//! The floating subtitle bar (PLAN §13.1, tasks 1.9 and 1.10).
//!
//! A frameless, transparent, always-on-top window holding two lines: the
//! source, and its translation. `li-core` does the work; this window shows it
//! and gets out of the way.
//!
//! Task 1.10 adds the three things that make it usable without a terminal: a
//! settings window that writes `config.toml`, a bar that stays where it was
//! dragged, and global hotkeys -- see [`Action`] for why they have to be
//! global.
//!
//! ## Probe modes
//!
//! `LI_PROBE=1` opens the bar with sample text, a checklist, and no engine.
//! Four of the window's properties are things a Wayland compositor is free to
//! refuse -- always-on-top especially, because Wayland has no protocol for a
//! client to ask -- and PLAN §19.7 has had them down as unverified since the
//! plan was written. Half a minute of looking answers it, and nothing above
//! this line is worth building until it is answered. (Answered on 2026-09-03:
//! `docs/phase1/TASK_1_9_WAYLAND.md`.)
//!
//! `LI_PROBE=replay` opens the real bar and feeds it a scripted minute of
//! subtitle, for the question §19.3 leaves open -- whether the in-place
//! overwrite disturbs reading. See [`probe`].
//!
//! ## Why plain HTML and no bundler
//!
//! PLAN §13.1 said Svelte + Vite. The bar is two `<div>`s and a `listen()`
//! call, and the settings window is a form; neither needs a component
//! framework, and skipping it keeps node out of the build entirely -- one
//! fewer toolchain to have installed on Windows and in CI. If the UI ever
//! grows enough to want components, `frontendDist` points at a directory and
//! can point at a built one instead.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod probe;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Result;
use li_audio::AudioSource;
use li_core::Engine;
use li_core::config::{BarPosition, EngineConfig, HotkeyCfg, UiCfg};
use li_core::download;
use li_core::models::Models;
use li_types::{DeviceInfo, UiEvent};
use tauri::{
    AppHandle, Emitter, Manager, PhysicalPosition, PhysicalSize, WebviewUrl, WebviewWindow,
    WebviewWindowBuilder, Window, WindowEvent,
};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};
use tokio::sync::Mutex;

const BAR: &str = "bar";
const SETTINGS: &str = "settings";
const DOWNLOAD: &str = "download";

/// The engine, when there is one. The probe modes run the bar without it.
struct State {
    engine: Arc<Mutex<Engine>>,
    paused: AtomicBool,
}

/// Everything about the bar that changes while it is open.
///
/// One lock and not a field each: a resize reads the settings and writes the
/// placement, a settings change writes the settings and re-places, and a drag
/// does both -- they are never independent.
struct Shell {
    /// What the file says, and what a save writes back. Never the settings
    /// window's unsaved preview: a drag saves the bar's position, and it must
    /// not carry a font size nobody agreed to into the file with it.
    cfg: EngineConfig,
    /// What the bar is actually showing -- `cfg.ui`, unless the settings
    /// window is previewing something that has not been saved.
    live: UiCfg,
    /// Content height in CSS pixels: estimated until the front end measures
    /// itself, and measured after that. See [`fit_bar`].
    height: f64,
    /// The webview's `devicePixelRatio`, which is not Tauri's scale factor.
    dpr: f64,
    /// The position this side last asked for, so that a `Moved` event saying
    /// anything else is the user dragging the bar.
    placed: Option<PhysicalPosition<i32>>,
    /// What `[ui]` was when the settings window opened, so that closing it
    /// without saving puts the bar back.
    ///
    /// On this side rather than in the page, because a window's close cannot
    /// be reliably held open from JavaScript long enough to send anything:
    /// measured, the settings window was destroyed while its revert was still
    /// on its way here, and the bar kept a font size nobody had agreed to.
    preview_base: Option<UiCfg>,
    /// Hotkeys the platform would not give us, for the settings window to
    /// show. A combination another application already holds fails silently
    /// otherwise, and looks exactly like a broken feature.
    hotkeys_refused: Vec<String>,
}

type Shared = std::sync::Mutex<Shell>;

fn shell(app: &AppHandle) -> tauri::State<'_, Shared> {
    app.state::<Shared>()
}

// --- commands ---------------------------------------------------------------

/// The bar's own settings (PLAN §14 `[ui]`).
///
/// Read by the front end at load: font size and opacity are CSS, and the row
/// budgets decide where a long line is clipped. The window geometry they also
/// determine is applied on this side, in [`reposition`].
#[tauri::command]
fn ui_config(app: AppHandle) -> UiCfg {
    shell(&app).lock().unwrap().live.clone()
}

/// The whole config, for the settings window to render.
#[tauri::command]
fn read_config(app: AppHandle) -> EngineConfig {
    shell(&app).lock().unwrap().cfg.clone()
}

#[tauri::command]
fn config_path() -> String {
    EngineConfig::path().display().to_string()
}

/// What this build is: the version from `Cargo.toml`, the commit from
/// `build.rs`.
///
/// Shown in the settings window and logged at startup. The version on its own
/// was not enough to tell two packages apart -- both said `0.1.0` -- and the
/// question "which build am I running" is always asked when something looks
/// wrong, which is the worst moment to have no answer.
const BUILD: &str = concat!(
    "LiveInterpreter ",
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("LI_BUILD_ID"),
    ")"
);

#[tauri::command]
fn build_id() -> &'static str {
    BUILD
}

/// Apply `[ui]` without writing it to disk.
///
/// So a slider can be dragged and the bar seen to change. Nothing is saved
/// until the settings window asks; closing it without saving leaves the file
/// as it was, but not the running bar -- the change stays until the next
/// start. That is the price of showing the change on the real bar rather than
/// on a preview of one, and the real bar is the only place the answer to "is
/// this readable" lives.
#[tauri::command]
fn preview_ui(app: AppHandle, ui: UiCfg) -> Result<(), String> {
    apply_ui(&app, ui).map_err(|e| e.to_string())
}

/// Write `config.toml` (PLAN §17 task 1.10).
///
/// `[ui]` is applied as it is edited, the hotkeys are re-registered here, and
/// the audio source reopens the capture device (see [`reopen_capture`]). The
/// rest -- the lanes, the transcript -- is still read only when the engine
/// starts, and this program has no way to swap a model out from under a
/// running pipeline (PLAN §10.3's `update_config` is not built). Saying so in
/// the window beats a setting that looks applied and is not.
#[tauri::command]
fn save_config(app: AppHandle, cfg: EngineConfig) -> Result<Saved, String> {
    apply_ui(&app, cfg.ui.clone()).map_err(|e| e.to_string())?;
    let (path, rebind, source_changed) = {
        let state = shell(&app);
        let mut s = state.lock().unwrap();
        let rebind = s.cfg.hotkeys != cfg.hotkeys;
        let source_changed = s.cfg.audio.source != cfg.audio.source;
        s.live = cfg.ui.clone();
        s.preview_base = Some(cfg.ui.clone());
        s.cfg = cfg;
        (
            s.cfg.save().map_err(|e| format!("{e:#}"))?,
            rebind,
            source_changed,
        )
    };
    if source_changed {
        reopen_capture(&app);
    }
    if rebind {
        // Unregister first: the old combination is still held by this process,
        // and a hotkey changed to one this program already owns would
        // otherwise be refused by its own previous binding.
        let _ = app.global_shortcut().unregister_all();
        let keys = shell(&app).lock().unwrap().cfg.hotkeys.clone();
        let refused = register_hotkeys(&app, &keys);
        shell(&app).lock().unwrap().hotkeys_refused = refused;
    }
    let state = shell(&app);
    let s = state.lock().unwrap();
    Ok(Saved {
        path: path.display().to_string(),
        hotkeys_refused: s.hotkeys_refused.clone(),
    })
}

/// Point the running engine at the audio source that was just saved.
///
/// The capture device is opened once, by `Engine::start`, and there is no way
/// to aim a running pipeline at a different one -- so until task 1.23 changing
/// the source in the settings window changed the file and nothing else, and
/// the bar went on transcribing whatever it had opened when it started. That
/// is what was reported: switching to the microphone kept the speakers. The
/// window did say 下次啟動生效, which is not the same as being right.
///
/// Restarting the engine is a heavy way to change one device -- the models
/// reload, which is seconds, and the transcript continues in a new file
/// because `stop` closes the old one properly. Both are worth it: the cost of
/// getting this wrong is a whole session recorded off the wrong input, and it
/// is a deliberate act by someone who is watching the bar when they do it.
///
/// Off the command's thread: the engine lock is held across a model load, and
/// `save_config` is what the settings window is waiting on.
fn reopen_capture(app: &AppHandle) {
    // The probe modes run the bar with no engine at all.
    let Some((engine, paused)) = app
        .try_state::<State>()
        .map(|s| (s.engine.clone(), s.paused.load(Ordering::Relaxed)))
    else {
        return;
    };
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let mut engine = engine.lock().await;
        if let Err(e) = engine.stop().await {
            // Worth the log, not worth refusing to open the new device: the
            // old session is over either way.
            tracing::warn!("stopping the engine to change the audio source: {e:#}");
        }
        match engine.start().await {
            Ok(()) => {
                // A restart begins unpaused, and the bar's own idea of paused
                // is in `State`. Left alone, the pause hotkey would need two
                // presses to bite again.
                engine.pause(paused);
                notice(&app, "音訊來源已切換");
            }
            Err(e) => {
                tracing::error!("opening the new audio source: {e:#}");
                notice(&app, "音訊來源打不開，請看設定視窗");
            }
        }
    });
}

/// What the settings window shows after a save: where it went, and which
/// hotkeys the desktop would not hand over.
#[derive(serde::Serialize)]
struct Saved {
    path: String,
    hotkeys_refused: Vec<String>,
}

/// Hotkeys that could not be registered, for the settings window.
#[tauri::command]
fn hotkey_problems(app: AppHandle) -> Vec<String> {
    shell(&app).lock().unwrap().hotkeys_refused.clone()
}

/// The inputs this machine offers, for the settings window's source list.
#[tauri::command]
fn audio_devices() -> Result<Vec<DeviceInfo>, String> {
    li_audio::DesktopSource::list_devices().map_err(|e| format!("{e:#}"))
}

#[tauri::command]
fn open_settings(app: AppHandle) -> Result<(), String> {
    settings_window(&app).map(|_| ()).map_err(|e| e.to_string())
}

/// Size the settings window to its own page.
///
/// Same measurement problem as [`fit_bar`], and the same answer: the window is
/// sized in screen pixels and the page is laid out in its own, which are 1.6
/// times smaller on this desktop. A settings window built at a plausible
/// number of Tauri's pixels comes out showing two thirds of the form.
#[tauri::command]
fn fit_settings(window: Window, width: f64, height: f64, dpr: f64) -> Result<(), String> {
    let dpr = if dpr.is_finite() && dpr > 0.0 {
        dpr
    } else {
        1.0
    };
    let (mut w, mut h) = ((width * dpr).round(), (height * dpr).round());
    // Never taller than the screen: the form grows with the number of audio
    // devices, and a window taller than the monitor cannot be closed on a
    // desktop that draws no title bar off-screen.
    if let Ok(Some(monitor)) = window.current_monitor() {
        let size = monitor.size();
        w = w.min(size.width as f64 * 0.9);
        h = h.min(size.height as f64 * 0.9);
    }
    window
        .set_size(PhysicalSize::new(w.max(320.0) as u32, h.max(240.0) as u32))
        .map_err(|e| e.to_string())?;
    window.center().map_err(|e| e.to_string())
}

/// Let clicks reach whatever is behind the bar (PLAN §13.1).
#[tauri::command]
fn set_click_through(app: AppHandle, on: bool) -> Result<(), String> {
    click_through(&app, on).map_err(|e| e.to_string())
}

/// Size the window to the content the front end just laid out.
///
/// The height is the one number this side cannot work out: it depends on the
/// font, on how many rows the translation wrapped to, and on whether the probe
/// checklist is there. So the front end measures and this places -- which keeps
/// every monitor coordinate in one function.
///
/// `dpr` comes with it, and is not optional. It is measured here that Tauri and
/// WebKitGTK do not agree about pixels: `Monitor::scale_factor()` says 1.0 and
/// the window really is sized in X pixels, while the webview lays the page out
/// at `devicePixelRatio` 1.6 from the desktop's font DPI. A bar sized to the
/// page's own CSS pixels therefore came out 1.6x too short and cut the
/// translation off below its own background. The front end reports both, and
/// everything on this side is in physical pixels.
#[tauri::command]
fn fit_bar(app: AppHandle, height: f64, dpr: f64) -> Result<(), String> {
    {
        let state = shell(&app);
        let mut s = state.lock().unwrap();
        // A front-end bug must not be able to ask for a full-screen or
        // zero-height always-on-top window, which is a thing the user would
        // have to kill from a terminal.
        s.height = height.clamp(40.0, 600.0);
        s.dpr = if dpr.is_finite() && dpr > 0.0 {
            dpr
        } else {
            1.0
        };
    }
    FITTED.store(true, Ordering::Relaxed);
    tracing::debug!(height, dpr, "the front end measured itself");
    reposition(&app).map_err(|e| e.to_string())
}

/// Whether the front end has ever measured itself. Only for the warning below.
static FITTED: AtomicBool = AtomicBool::new(false);
/// Counts `Moved` events, so only the last one of a drag is acted on.
static MOVES: AtomicU64 = AtomicU64::new(0);

#[tauri::command]
async fn pause(state: tauri::State<'_, State>, on: bool) -> Result<(), String> {
    state.engine.lock().await.pause(on);
    state.paused.store(on, Ordering::Relaxed);
    Ok(())
}

/// Where this session's transcript is being written (PLAN §13.1).
#[tauri::command]
async fn transcripts(state: tauri::State<'_, State>) -> Result<Vec<String>, String> {
    Ok(state
        .engine
        .lock()
        .await
        .transcripts()
        .iter()
        .map(|p| p.display().to_string())
        .collect())
}

/// End the program.
///
/// There has to be a button for this. The bar draws no decorations and is kept
/// off the taskbar, so until 1.14 put a package on a real machine nobody had
/// noticed that the only way to stop it was `pkill` from a terminal.
///
/// `Engine::stop().await` and not a bare `app.exit`: stopping drops the capture
/// source and then waits for the sink to close the transcript files. Quitting
/// must not mean losing the last lines of a session. The probe modes have no
/// engine to stop and go straight to the exit.
#[tauri::command]
async fn quit(app: AppHandle) -> Result<(), String> {
    // The state guard is not held across the await: `tauri::State` borrows the
    // handle, and a future holding one is not `Send`.
    let engine = app.try_state::<State>().map(|s| s.engine.clone());
    if let Some(engine) = engine
        && let Err(e) = engine.lock().await.stop().await
    {
        // Losing the tail of a transcript is worth a line in the log, but not
        // worth refusing to quit: the person has asked to leave.
        tracing::warn!("stopping the engine on the way out: {e:#}");
    }
    app.exit(0);
    Ok(())
}

// --- geometry ---------------------------------------------------------------

/// Where a bar of this height goes: against the top or bottom edge of the
/// monitor, centred, plus wherever the user dragged it.
///
/// The anchored edge is the one that stays put. A translation that wraps to a
/// second row makes the bar taller, and the bar has to grow *away* from the
/// screen edge -- growing towards it would walk the whole bar off the desktop
/// one wrapped sentence at a time. That is the property worth a test.
///
/// `origin` and `screen` are the monitor's, in physical pixels; `height` and
/// `ui.margin_px` are CSS pixels, which `dpr` converts. Physical pixels are the
/// only unit both sides agree on -- see [`fit_bar`]. The monitor rather than
/// the desktop, too: on this three-screen machine, centring on the desktop puts
/// the bar across a bezel.
fn geometry(
    origin: (f64, f64),
    screen: (f64, f64),
    ui: &UiCfg,
    height: f64,
    dpr: f64,
) -> (PhysicalSize<u32>, PhysicalPosition<i32>) {
    let (screen_w, screen_h) = screen;

    let width = (screen_w * ui.width_pct.clamp(10.0, 100.0) / 100.0).round();
    let height = (height * dpr).round();
    let margin = ui.margin_px * dpr;

    let x = origin.0 + (screen_w - width) / 2.0 + ui.offset_x;
    let y = match ui.position {
        BarPosition::Bottom => origin.1 + screen_h - height - margin + ui.offset_y,
        BarPosition::Top => origin.1 + margin + ui.offset_y,
    };
    // The offset is remembered, so it has to be survivable: it may have been
    // saved on a bigger screen, or come from a drag that put half the bar past
    // the edge. Either way the bar stays somewhere it can be read and grabbed.
    let x = on_screen(x, origin.0, screen_w, width);
    let y = on_screen(y, origin.1, screen_h, height);
    (
        PhysicalSize::new(width as u32, height as u32),
        PhysicalPosition::new(x as i32, y as i32),
    )
}

fn on_screen(v: f64, origin: f64, screen: f64, size: f64) -> f64 {
    v.clamp(origin, (origin + screen - size).max(origin))
}

/// Put the bar where the current settings and the last measurement say.
fn reposition(app: &AppHandle) -> tauri::Result<()> {
    let Some(window) = app.get_webview_window(BAR) else {
        return Ok(());
    };
    let (ui, height, dpr) = {
        let state = shell(app);
        let s = state.lock().unwrap();
        (s.live.clone(), s.height, s.dpr)
    };
    let Some((size, position)) = monitor_geometry(&window, &ui, height, dpr)? else {
        return Ok(());
    };
    window.set_size(size)?;
    // After the resize, not before: a window manager is free to re-anchor a
    // resize to the top left corner, which would undo the position.
    window.set_position(position)?;
    shell(app).lock().unwrap().placed = Some(position);
    Ok(())
}

fn monitor_geometry(
    window: &WebviewWindow,
    ui: &UiCfg,
    height: f64,
    dpr: f64,
) -> tauri::Result<Option<(PhysicalSize<u32>, PhysicalPosition<i32>)>> {
    let Some(monitor) = window.current_monitor()?.or(window.primary_monitor()?) else {
        // Not a reason to fail: the compositor put the window somewhere, and
        // there it stays.
        tracing::warn!("no monitor reported: leaving the bar where it was placed");
        return Ok(None);
    };
    let (origin, screen) = (monitor.position(), monitor.size());
    Ok(Some(geometry(
        (origin.x as f64, origin.y as f64),
        (screen.width as f64, screen.height as f64),
        ui,
        height,
        dpr,
    )))
}

/// Roughly how tall the bar will turn out, in CSS pixels, for the first frame.
///
/// The front end measures the real height and sends it back within a frame, so
/// this only has to be close enough that the window is not visibly the wrong
/// size while the page loads. Waiting for the measurement instead is what the
/// first attempt did, by building the window hidden -- and a hidden WebKitGTK
/// window runs no `requestAnimationFrame` callbacks, so it never measured, so
/// it was never shown.
fn estimate_height(ui: &UiCfg) -> f64 {
    const PADDING: f64 = 22.0; // 10 px each side, plus the gap between the rows
    const LINE: f64 = 1.35; // the rows' `line-height`
    const SOURCE: f64 = 0.8; // ...and the source row's smaller `font-size`
    let rows = SOURCE * ui.source_rows.max(1) as f64 + 1.0;
    PADDING + (ui.font_size * LINE * rows).round()
}

// --- dragging ---------------------------------------------------------------

/// A `Moved` event this side did not cause is the user dragging the bar.
///
/// It has to be remembered, and not just left alone, because the window is
/// re-placed every time the translation wraps to another row -- without this
/// the bar would jump back to the middle of the screen several times a minute,
/// which is exactly the "跳行" task 1.9 was about, one level up.
///
/// The drag is recorded as an offset from where the settings would have put
/// the bar, so it survives a resolution change and a different monitor.
fn on_moved(app: &AppHandle, to: PhysicalPosition<i32>) {
    let ours = shell(app).lock().unwrap().placed;
    // A window manager may answer a `set_position` a pixel or two off; that is
    // not a drag.
    if ours.is_some_and(|p| (p.x - to.x).abs() <= 2 && (p.y - to.y).abs() <= 2) {
        return;
    }
    // X11 reports a move per frame while the button is held. Only the last one
    // is the answer, so each schedules itself and the stale ones stand down.
    let generation = MOVES.fetch_add(1, Ordering::Relaxed) + 1;
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(Duration::from_millis(400)).await;
        if MOVES.load(Ordering::Relaxed) != generation {
            return;
        }
        if let Err(e) = settle(&app, to) {
            tracing::warn!("recording the drag: {e:#}");
        }
    });
}

fn settle(app: &AppHandle, to: PhysicalPosition<i32>) -> Result<()> {
    let Some(window) = app.get_webview_window(BAR) else {
        return Ok(());
    };
    let (mut ui, height, dpr) = {
        let state = shell(app);
        let s = state.lock().unwrap();
        (s.live.clone(), s.height, s.dpr)
    };
    // Where it would have gone with no offset at all: the difference is what
    // the user asked for by dragging.
    ui.offset_x = 0.0;
    ui.offset_y = 0.0;
    let Some((_, anchor)) = monitor_geometry(&window, &ui, height, dpr)? else {
        return Ok(());
    };
    let (dx, dy) = ((to.x - anchor.x) as f64, (to.y - anchor.y) as f64);

    {
        let state = shell(app);
        let mut s = state.lock().unwrap();
        s.live.offset_x = dx;
        s.live.offset_y = dy;
        // Into the saved config as well, and only these two fields: this
        // writes the file, and the settings window may be previewing a font
        // size the user has not agreed to keep.
        s.cfg.ui.offset_x = dx;
        s.cfg.ui.offset_y = dy;
    }
    // Re-place rather than leave it: this is what clamps a bar dragged half
    // off the screen back onto it, and the user sees it happen while their
    // hand is still on the mouse.
    reposition(app)?;
    let path = shell(app).lock().unwrap().cfg.save()?;
    tracing::info!(dx, dy, path = %path.display(), "the bar was moved");
    notice(app, "位置已記住");
    Ok(())
}

// --- hotkeys ----------------------------------------------------------------

/// What a global hotkey does (PLAN §17 task 1.10).
///
/// Global, and not a button on the bar, because two of the three are otherwise
/// unreachable. A bar with click-through on receives no clicks, so nothing
/// drawn on it can turn click-through off again -- PLAN §13.1's "switch it off
/// while the pointer is over the bar" cannot be done at all, since a window
/// that ignores the cursor stops being told the cursor arrived. And pausing is
/// wanted at the moment something private is about to be said, which is not a
/// moment to go looking for a window.
#[derive(Clone, Copy, Debug)]
enum Action {
    ClickThrough,
    Pause,
    Settings,
}

impl Action {
    fn run(self, app: &AppHandle) {
        let done = match self {
            Action::ClickThrough => {
                let on = !shell(app).lock().unwrap().live.click_through;
                click_through(app, on)
            }
            Action::Settings => settings_window(app).map(|_| ()),
            Action::Pause => return toggle_pause(app),
        };
        if let Err(e) = done {
            tracing::warn!("hotkey {self:?}: {e}");
        }
    }
}

fn click_through(app: &AppHandle, on: bool) -> tauri::Result<()> {
    if let Some(window) = app.get_webview_window(BAR) {
        window.set_ignore_cursor_events(on)?;
    }
    let changed = {
        let state = shell(app);
        let mut s = state.lock().unwrap();
        let changed = s.live.click_through != on;
        s.live.click_through = on;
        s.cfg.ui.click_through = on;
        changed
    };
    // Saved, so a bar that has found its place is out of the way again on the
    // next run without being told twice.
    if changed && let Err(e) = shell(app).lock().unwrap().cfg.save() {
        tracing::warn!("saving click-through: {e:#}");
    }
    notice(
        app,
        if on {
            "點擊穿透 開（滑鼠會穿過字幕條）"
        } else {
            "點擊穿透 關（可拖曳）"
        },
    );
    Ok(())
}

fn toggle_pause(app: &AppHandle) {
    let Some(state) = app.try_state::<State>() else {
        // The probe modes have no engine, and saying so beats a hotkey that
        // looks broken.
        notice(app, "沒有引擎可暫停（probe 模式）");
        return;
    };
    let on = !state.paused.load(Ordering::Relaxed);
    state.paused.store(on, Ordering::Relaxed);
    notice(app, if on { "已暫停" } else { "繼續" });
    let engine = state.engine.clone();
    tauri::async_runtime::spawn(async move { engine.lock().await.pause(on) });
}

/// What the first run has to fetch, for the page that asks permission.
#[derive(serde::Serialize)]
struct PlanInfo {
    models: Vec<String>,
    total_bytes: u64,
    dir: String,
}

#[tauri::command]
fn model_plan(app: AppHandle) -> Result<PlanInfo, String> {
    let cfg = shell(&app).lock().unwrap().cfg.clone();
    let models = Models::new();
    let plan = download::plan(&models, &cfg).map_err(|e| format!("{e:#}"))?;
    Ok(PlanInfo {
        models: plan.models().into_iter().map(str::to_owned).collect(),
        total_bytes: plan.total_bytes,
        dir: models.root().display().to_string(),
    })
}

/// Fetch the missing models, then start the engine.
///
/// Starting it here rather than making the page ask separately keeps the one
/// invariant this is all for: the engine is started exactly once, and only
/// after its models are on disk.
#[tauri::command]
async fn fetch_models(app: AppHandle, state: tauri::State<'_, State>) -> Result<(), String> {
    let cfg = shell(&app).lock().unwrap().cfg.clone();
    let models = Models::new();
    let plan = download::plan(&models, &cfg).map_err(|e| format!("{e:#}"))?;

    // Emitted rather than returned: a 1 GB transfer has to show progress while
    // it runs, and a command's return value arrives only at the end.
    let emitter = app.clone();
    let total = plan.total_bytes;
    download::fetch(&plan, move |p| {
        let (stage, name, done) = match p {
            download::Progress::Started { name, .. } => ("fetching", name, None),
            download::Progress::Bytes { done, .. } => ("fetching", String::new(), Some(done)),
            download::Progress::Verifying { name } => ("checking", name, None),
            download::Progress::Unpacking { name } => ("unpacking", name, None),
            download::Progress::Finished { name } => ("done", name, None),
        };
        let _ = emitter.emit_to(
            DOWNLOAD,
            "models://progress",
            serde_json::json!({ "stage": stage, "name": name, "done": done, "total": total }),
        );
    })
    .await
    .map_err(|e| format!("{e:#}"))?;

    if let Err(e) = state.engine.lock().await.start().await {
        return Err(format!("模型下載完成，但引擎啟動失敗：{e:#}"));
    }
    if let Some(w) = app.get_webview_window(DOWNLOAD) {
        let _ = w.close();
    }
    Ok(())
}

fn download_window(app: &AppHandle) -> tauri::Result<WebviewWindow> {
    if let Some(window) = app.get_webview_window(DOWNLOAD) {
        window.show()?;
        window.set_focus()?;
        return Ok(window);
    }
    WebviewWindowBuilder::new(app, DOWNLOAD, WebviewUrl::App("download.html".into()))
        .title("LiveInterpreter — 下載模型")
        .inner_size(560.0, 420.0)
        .always_on_top(true)
        .build()
}

fn settings_window(app: &AppHandle) -> tauri::Result<WebviewWindow> {
    {
        // Only on the way in: opening the window again while it is already
        // open must not make the current preview the thing to revert to.
        let state = shell(app);
        let mut s = state.lock().unwrap();
        if s.preview_base.is_none() {
            s.preview_base = Some(s.live.clone());
        }
    }
    if let Some(window) = app.get_webview_window(SETTINGS) {
        window.show()?;
        window.unminimize()?;
        window.set_focus()?;
        return Ok(window);
    }
    // Always on top as well: the bar is, and a settings window the bar covers
    // is a settings window whose save button is under a subtitle.
    WebviewWindowBuilder::new(app, SETTINGS, WebviewUrl::App("settings.html".into()))
        .title("LiveInterpreter 設定")
        .inner_size(560.0, 720.0)
        .always_on_top(true)
        .build()
}

/// A line of text on the bar for a moment: what a hotkey just did.
///
/// A global hotkey is pressed while looking at something else, and gives no
/// sign it arrived. Without this, a click-through toggle that failed to
/// register and one that worked look identical.
fn notice(app: &AppHandle, text: &str) {
    let _ = app.emit_to(BAR, "ui://notice", text);
}

/// The settings window is gone; put back whatever it was previewing.
///
/// A save takes the previewed values as the new baseline, so this only ever
/// undoes changes nobody kept.
fn revert_preview(app: &AppHandle) {
    let base = shell(app).lock().unwrap().preview_base.take();
    let Some(base) = base else { return };
    if base == shell(app).lock().unwrap().live {
        return;
    }
    tracing::debug!("the settings window closed unsaved: putting the bar back");
    if let Err(e) = apply_ui(app, base) {
        tracing::warn!("reverting the preview: {e}");
    }
}

fn apply_ui(app: &AppHandle, ui: UiCfg) -> tauri::Result<()> {
    let was = {
        let state = shell(app);
        let mut s = state.lock().unwrap();
        std::mem::replace(&mut s.live, ui.clone())
    };
    if was.click_through != ui.click_through {
        click_through(app, ui.click_through)?;
    }
    // The front end restyles itself and measures again; `fit_bar` then places
    // the window at the height that came out. This call is for the settings
    // that change the window without changing the page -- width and margin.
    let _ = app.emit_to(BAR, "ui://config", &ui);
    reposition(app)
}

fn register_hotkeys(app: &AppHandle, keys: &HotkeyCfg) -> Vec<String> {
    let mut refused = Vec::new();
    for (combo, action) in [
        (&keys.click_through, Action::ClickThrough),
        (&keys.pause, Action::Pause),
        (&keys.settings, Action::Settings),
    ] {
        let combo = combo.trim();
        if combo.is_empty() {
            continue;
        }
        let result = app.global_shortcut().on_shortcut(combo, move |app, _, e| {
            // Pressed only. Acting on the release as well toggles twice.
            if e.state == ShortcutState::Pressed {
                action.run(app);
            }
        });
        match result {
            Ok(()) => tracing::info!(combo, ?action, "hotkey registered"),
            Err(e) => {
                // Either the combination will not parse or something else on
                // the desktop already holds it. Both end as a key that does
                // nothing, which is why this is not left to the log alone.
                tracing::warn!(combo, ?action, "hotkey refused: {e}");
                refused.push(format!("{combo} ({action:?}): {e}"));
            }
        }
    }
    refused
}

// --- startup ----------------------------------------------------------------

/// Ask GTK for the X11 backend before it initialises (Linux only).
///
/// Measured on KDE Plasma: on native Wayland a window opened after the bar
/// covers it, because Wayland has no protocol for a client to request
/// always-on-top -- xdg-shell has no such request, and the flag Tauri stores
/// is never sent anywhere. Through XWayland, KWin honours
/// `_NET_WM_STATE_ABOVE` and the bar stays on top even of native Wayland
/// windows raised after it. Until the bar moves to `wlr-layer-shell`, which
/// is the protocol actually meant for this and which a GTK3/webkit window
/// cannot use, XWayland is the path where the bar is a bar.
///
/// The escape hatch is `LI_GDK_BACKEND`, not `GDK_BACKEND`: a KDE Wayland
/// session exports `GDK_BACKEND=wayland` for every process it starts, so
/// deferring to that variable means deferring to a session default that knows
/// nothing about this window's requirements, and the bar silently loses the
/// one property it exists for. `LI_GDK_BACKEND=wayland` asks for the Wayland
/// path deliberately.
// The workspace denies `unsafe_code`; `set_var` is the only way to reach GTK,
// which reads this variable during its own initialisation and offers no API.
#[allow(unsafe_code)]
#[cfg(target_os = "linux")]
fn prefer_x11() {
    let want = std::env::var("LI_GDK_BACKEND").unwrap_or_else(|_| "x11".into());
    // SAFETY: single-threaded, first thing in `main`, before GTK or any
    // thread that might read the environment concurrently.
    unsafe { std::env::set_var("GDK_BACKEND", &want) };
}

/// What the window is for this run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// The bar, driven by the engine.
    Engine,
    /// The bar with the Wayland checklist over it, and no engine (§19.7).
    Checklist,
    /// The bar, driven by a script instead of by audio (§19.3).
    Replay,
}

impl Mode {
    fn from_env() -> Self {
        match std::env::var("LI_PROBE").as_deref() {
            Ok("1") => Mode::Checklist,
            Ok("replay") => Mode::Replay,
            _ => Mode::Engine,
        }
    }

    /// The front end is one page; the query string says which job it has.
    fn url(self) -> String {
        match self {
            Mode::Checklist => {
                // `LI_PROBE_CT=1` starts the checklist with click-through
                // already on. Once it works the toggle button cannot be
                // clicked either, so the only way to measure the property is
                // to not need the mouse.
                let ct = match std::env::var("LI_PROBE_CT").as_deref() {
                    Ok("1") => "&ct=1",
                    Ok("delay") => "&ct=delay",
                    _ => "",
                };
                format!("index.html?probe=1{ct}")
            }
            Mode::Engine | Mode::Replay => "index.html".into(),
        }
    }
}

fn main() {
    #[cfg(target_os = "linux")]
    prefer_x11();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("LI_LOG")
                // The bin target is `liveinterpreter-desktop`, so its tracing
                // target is `liveinterpreter_desktop` -- `li_desktop` matched
                // nothing and hid the line saying which backend GTK picked.
                .unwrap_or_else(|_| {
                    // `li_audio` is in here because two of the things that
                    // decide how late the bar is -- which device was actually
                    // opened, and how big its buffer is -- are only said once,
                    // at `info`, and without them a report of "it feels slow"
                    // has nothing to attach to.
                    "warn,li_core=info,li_asr=info,li_audio=info,liveinterpreter_desktop=info"
                        .into()
                }),
        )
        .init();

    // First line in the log, always: a report of what the bar did means
    // nothing until it is attached to a build.
    tracing::info!("{BUILD}");

    let mode = Mode::from_env();

    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            set_click_through,
            ui_config,
            read_config,
            config_path,
            build_id,
            preview_ui,
            save_config,
            audio_devices,
            hotkey_problems,
            open_settings,
            fit_settings,
            fit_bar,
            pause,
            transcripts,
            model_plan,
            fetch_models,
            quit
        ])
        .on_window_event(|window, event| {
            let app = window.app_handle().clone();
            match (window.label(), event) {
                (BAR, WindowEvent::Moved(to)) => on_moved(&app, *to),
                (SETTINGS, WindowEvent::Destroyed) => revert_preview(&app),
                _ => {}
            }
        })
        .setup(move |app| {
            // Read before the window is built, and in every mode: a malformed
            // config is a startup error with a line number, not a bar that
            // quietly ignores the settings someone edited.
            let cfg = EngineConfig::load()?;
            let ui = cfg.ui.clone();
            let hotkeys = cfg.hotkeys.clone();
            app.manage(std::sync::Mutex::new(Shell {
                live: ui.clone(),
                height: estimate_height(&ui),
                // A guess at the webview's scale, which is not knowable until
                // the page has loaded. `fit_bar` replaces both with
                // measurements a frame later.
                dpr: 1.0,
                cfg,
                placed: None,
                preview_base: None,
                hotkeys_refused: Vec::new(),
            }));

            let window = WebviewWindowBuilder::new(app, BAR, WebviewUrl::App(mode.url().into()))
                .title("LiveInterpreter")
                .decorations(false)
                .transparent(true)
                .always_on_top(true)
                .skip_taskbar(true)
                .shadow(false)
                .resizable(true)
                // Placed while it is hidden, so the bar does not appear in
                // the middle of the screen and then jump. Shown from here
                // rather than from the front end: a window that waits to be
                // shown until the page measures itself is a window that
                // stays invisible when the page has a bug.
                .visible(false)
                .build()?;

            let handle = app.handle().clone();
            reposition(&handle)?;
            window.show()?;
            if ui.click_through {
                click_through(&handle, true)?;
            }

            let refused = register_hotkeys(&handle, &hotkeys);
            if !refused.is_empty() {
                notice(&handle, "有快捷鍵註冊失敗，見設定視窗");
                shell(&handle).lock().unwrap().hotkeys_refused = refused;
            }

            // A front end that dies before its first measurement leaves a bar
            // at the estimated height, which looks close enough to right to go
            // unnoticed. Task 1.9's first probe reported three Wayland failures
            // that were really a front end that never ran; this says so.
            let waiting = window.clone();
            tauri::async_runtime::spawn(async move {
                tokio::time::sleep(Duration::from_secs(2)).await;
                if !waiting.is_visible().unwrap_or(false) {
                    return;
                }
                if !FITTED.load(Ordering::Relaxed) {
                    tracing::warn!(
                        "the front end never called fit_bar: the bar is at its estimated size"
                    );
                }
            });

            // What the platform actually did, not what was asked for. On
            // Wayland `always_on_top` is a request nobody is obliged to honour.
            tracing::info!(
                on_top = window.is_always_on_top().unwrap_or(false),
                backend = std::env::var("GDK_BACKEND").unwrap_or_else(|_| "(default)".into()),
                session = std::env::var("XDG_SESSION_TYPE").unwrap_or_default(),
                "bar window created"
            );

            match mode {
                Mode::Checklist => {
                    tracing::info!("probe mode: the engine is not started");
                    return Ok(());
                }
                Mode::Replay => {
                    tracing::info!("replay mode: the engine is not started");
                    probe::spawn(app.handle().clone());
                    return Ok(());
                }
                Mode::Engine => {}
            }

            let cfg = shell(&handle).lock().unwrap().cfg.clone();
            let engine = Arc::new(Mutex::new(Engine::new(cfg)?));
            app.manage(State {
                engine: engine.clone(),
                paused: AtomicBool::new(false),
            });

            // Subscribed before `start`, so model-loading status reaches the bar.
            let mut events = {
                let e = engine.blocking_lock();
                e.subscribe()
            };
            let emitter = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                loop {
                    match events.recv().await {
                        Ok(ev) => {
                            let _ = emitter.emit("engine://event", UiEvent::from(&ev));
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!("the bar fell {n} events behind");
                        }
                        Err(_) => break,
                    }
                }
            });

            // A first run has no models, and `start` would fail on the first
            // one it could not resolve -- which used to be the whole story
            // (task 1.13). Ask before starting, and if anything is missing open
            // the download window instead; `fetch_models` starts the engine
            // when it is done.
            let cfg_for_plan = shell(&handle).lock().unwrap().cfg.clone();
            match download::plan(&Models::new(), &cfg_for_plan) {
                Ok(plan) if !plan.is_empty() => {
                    tracing::info!(
                        models = ?plan.models(),
                        mib = plan.total_bytes / (1024 * 1024),
                        "models are missing; opening the download window instead of starting"
                    );
                    download_window(&handle)?;
                    return Ok(());
                }
                Ok(_) => {}
                // A manifest or config problem is not a reason to refuse to
                // start: `start` will report whatever is actually wrong, in the
                // words of the thing that is wrong.
                Err(e) => tracing::warn!("checking for models: {e:#}"),
            }

            tauri::async_runtime::spawn(async move {
                if let Err(e) = engine.lock().await.start().await {
                    tracing::error!("starting the engine: {e:#}");
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("running the Tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2880x1800 monitor that is not the first one, so a bar placed with
    /// screen coordinates instead of monitor coordinates fails the test.
    const ORIGIN: (f64, f64) = (2624.0, 2304.0);
    const SCREEN: (f64, f64) = (2880.0, 1800.0);

    fn ui() -> UiCfg {
        UiCfg::default()
    }

    #[test]
    fn the_bar_sits_above_the_bottom_edge_of_its_own_monitor() {
        let (size, at) = geometry(ORIGIN, SCREEN, &ui(), 76.0, 1.0);
        assert_eq!(size.width, 2304, "80% of the monitor");
        assert_eq!(at.x, 2912, "centred on the monitor, not on the desktop");
        assert_eq!(at.y as f64 + size.height as f64, 2304.0 + 1800.0 - 40.0);
    }

    /// The property the whole layout rests on: the bar grows away from the edge
    /// it is anchored to. Growing towards it walks the bar off the screen one
    /// wrapped translation at a time.
    #[test]
    fn a_taller_bar_keeps_its_bottom_edge_and_moves_its_top() {
        let short = geometry(ORIGIN, SCREEN, &ui(), 76.0, 1.0);
        let tall = geometry(ORIGIN, SCREEN, &ui(), 106.0, 1.0);
        let bottom = |(s, p): (PhysicalSize<u32>, PhysicalPosition<i32>)| p.y + s.height as i32;
        assert_eq!(bottom(short), bottom(tall));
        assert!(tall.1.y < short.1.y);
    }

    #[test]
    fn the_top_anchor_grows_the_other_way() {
        let mut cfg = ui();
        cfg.position = BarPosition::Top;
        let short = geometry(ORIGIN, SCREEN, &cfg, 76.0, 1.0);
        let tall = geometry(ORIGIN, SCREEN, &cfg, 106.0, 1.0);
        assert_eq!(short.1.y, tall.1.y, "the top edge is the one that stays");
        assert_eq!(short.1.y as f64, ORIGIN.1 + cfg.margin_px);
    }

    /// The bug this file's `fit_bar` exists to prevent: WebKitGTK laid the page
    /// out at 1.6 CSS pixels to the screen pixel while Tauri reported 1.0, and
    /// a window sized in the page's pixels cut the translation off.
    #[test]
    fn a_scaled_webview_gets_a_window_in_screen_pixels() {
        let (size, _) = geometry(ORIGIN, SCREEN, &ui(), 76.0, 1.6);
        assert_eq!(size.height, 122);
        // ...and the margin scales with it, or the gap under the bar shrinks to
        // a third of what it looks like on a 1x screen.
        let (_, at) = geometry(ORIGIN, SCREEN, &ui(), 76.0, 1.6);
        assert_eq!(at.y as f64 + 122.0, 2304.0 + 1800.0 - 40.0 * 1.6);
    }

    #[test]
    fn a_full_width_bar_is_still_on_the_monitor() {
        let mut cfg = ui();
        cfg.width_pct = 100.0;
        let (size, at) = geometry(ORIGIN, SCREEN, &cfg, 76.0, 1.0);
        assert_eq!((at.x, size.width), (2624, 2880));
    }

    /// Task 1.10: a dragged bar keeps the place it was dragged to when the
    /// translation wraps and the window is re-placed -- and still grows away
    /// from the edge, so the two rules do not fight.
    #[test]
    fn a_dragged_bar_stays_where_it_was_put_when_it_grows() {
        let mut cfg = ui();
        // Within the slack a 80%-wide bar has: further left than this and the
        // clamp below takes over, which is the next test.
        cfg.offset_x = -200.0;
        cfg.offset_y = -200.0;
        let short = geometry(ORIGIN, SCREEN, &cfg, 76.0, 1.0);
        let plain = geometry(ORIGIN, SCREEN, &ui(), 76.0, 1.0);
        assert_eq!(short.1.x, plain.1.x - 200);
        assert_eq!(short.1.y, plain.1.y - 200);

        let tall = geometry(ORIGIN, SCREEN, &cfg, 106.0, 1.0);
        let bottom = |(s, p): (PhysicalSize<u32>, PhysicalPosition<i32>)| p.y + s.height as i32;
        assert_eq!(bottom(short), bottom(tall), "still bottom-anchored");
    }

    /// An offset saved on a wider screen, or a drag that threw the bar off the
    /// edge, must not leave it somewhere it cannot be read or grabbed.
    #[test]
    fn an_offset_from_another_screen_is_pulled_back_on() {
        let mut cfg = ui();
        cfg.offset_x = 9000.0;
        cfg.offset_y = 9000.0;
        let (size, at) = geometry(ORIGIN, SCREEN, &cfg, 76.0, 1.0);
        assert_eq!(at.x, (ORIGIN.0 + SCREEN.0) as i32 - size.width as i32);
        assert_eq!(at.y, (ORIGIN.1 + SCREEN.1) as i32 - size.height as i32);
        let mut cfg = ui();
        cfg.offset_x = -9000.0;
        cfg.offset_y = -9000.0;
        let (_, at) = geometry(ORIGIN, SCREEN, &cfg, 76.0, 1.0);
        assert_eq!((at.x, at.y), (ORIGIN.0 as i32, ORIGIN.1 as i32));
    }
}
