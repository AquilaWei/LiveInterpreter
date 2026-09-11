// The settings window (PLAN §14, §17 task 1.10).
//
// It edits the same `config.toml` a person edits by hand, so it writes through
// `li_core::config`, which merges into the existing document rather than
// serialising over it -- comments and the sections this program does not read
// yet both survive.
//
// ## Two kinds of setting
//
// `[ui]` is applied to the running bar as the slider moves: the only place the
// question "is this readable at this size, from where I sit" can be answered
// is on the real bar over the real screen. The audio source is applied on save
// too, by restarting the engine (task 1.23) -- a device that was switched and
// then quietly ignored is the whole reason that is worth the model reload.
// The transcript is still read only when the engine starts, and there is no
// way to swap a model out from under a running pipeline (PLAN §10.3's
// `update_config` is not built). That one says "下次啟動生效" rather than
// pretending.
//
// Closing without saving puts the live settings back where they were, because
// a preview that sticks is not a preview.

const { invoke } = window.__TAURI__.core;
const win = window.__TAURI__.window.getCurrentWindow();

const $ = (id) => document.getElementById(id);

// `[ui]` fields, by the id of the control that edits them. The `output`
// beside a range shows the value, so the slider is not a mystery.
const UI_NUMBERS = [
  "font_size", "opacity", "width_pct", "margin_px",
  "source_rows", "target_rows", "min_dwell_ms",
];
const UI_FLAGS = ["show_source", "click_through"];
const FORMATS = ["txt", "jsonl", "srt", "vtt"];

let cfg = null;      // the whole config, as read
let original = null; // `[ui]` as it was on open, for a close without saving
let saved = false;

function readUi() {
  const ui = { ...cfg.ui };
  for (const id of UI_NUMBERS) ui[id] = Number($(id).value);
  for (const id of UI_FLAGS) ui[id] = $(id).checked;
  ui.position = $("position").value;
  return ui;
}

function showValues() {
  for (const id of UI_NUMBERS) {
    const out = document.querySelector(`output[for=${id}]`);
    if (out) out.textContent = $(id).value;
  }
}

// Every `[ui]` control does the same thing: show the number, put it on the bar.
function live() {
  showValues();
  cfg.ui = readUi();
  invoke("preview_ui", { ui: cfg.ui }).catch((e) => say(`套用失敗：${e}`));
}

function say(text) {
  // The path is elided to keep the buttons their own size, so the whole of it
  // has to be somewhere: the tooltip.
  $("said").textContent = text;
  $("said").title = text;
}

async function load() {
  cfg = await invoke("read_config");
  original = { ...cfg.ui };

  for (const id of UI_NUMBERS) $(id).value = cfg.ui[id];
  for (const id of UI_FLAGS) $(id).checked = cfg.ui[id];
  $("position").value = cfg.ui.position;
  showValues();

  $("hk_click_through").value = cfg.hotkeys.click_through;
  $("hk_pause").value = cfg.hotkeys.pause;
  $("hk_settings").value = cfg.hotkeys.settings;
  problems(await invoke("hotkey_problems"));

  $("tr_enabled").checked = cfg.transcript.enabled;
  $("tr_bilingual").checked = cfg.transcript.bilingual_file;
  $("tr_dir").value = cfg.transcript.dir;
  $("formats").innerHTML = FORMATS.map((f) =>
    `<label><input type="checkbox" id="fmt_${f}" /> ${f}</label>`).join("");
  for (const f of FORMATS) $(`fmt_${f}`).checked = cfg.transcript.formats.includes(f);

  await sources();
  $("path").textContent = await invoke("config_path");
  // Which build this is. The version alone was never enough to tell two
  // packages apart, so the commit is in there too (`build.rs`).
  $("build").textContent = await invoke("build_id");

  for (const id of [...UI_NUMBERS, ...UI_FLAGS, "position"]) {
    $(id).addEventListener("input", live);
  }
  fit();
}

// The device list, with the two selectors that are not a device at the top.
// A source saved in the config that this machine cannot offer today stays in
// the list rather than silently becoming something else -- the headset may be
// plugged in again tomorrow.
async function sources() {
  const select = $("source");
  let devices = [];
  try {
    devices = await invoke("audio_devices");
  } catch (e) {
    say(`列不出音訊裝置：${e}`);
  }
  const options = [
    ["system", "系統輸出（這台機器正在播的聲音）"],
    ["mic", "預設麥克風"],
    // 「monitor」是 PulseAudio 的講法，Windows 上這一項其實是輸出端點的
    // loopback。標記講功能不講機制，兩邊才會是同一句話。
    ...devices.map((d) => [d.id, `${d.name}${d.is_loopback ? "（播放中的聲音）" : ""}`]),
  ];
  if (!options.some(([id]) => id === cfg.audio.source)) {
    options.push([cfg.audio.source, `${cfg.audio.source}（目前設定，現在找不到）`]);
  }
  select.innerHTML = "";
  for (const [id, name] of options) {
    const o = document.createElement("option");
    o.value = id;
    o.textContent = name;
    select.append(o);
  }
  select.value = cfg.audio.source;
}

function problems(list) {
  const el = $("hkfail");
  el.hidden = list.length === 0;
  el.textContent = list.length
    ? `這些快捷鍵註冊不起來（多半是桌面環境或別的程式先佔走了）：\n${list.join("\n")}`
    : "";
}

// Ctrl+S saves and Escape closes, because this is a form and those are what
// hands do to a form. (They are also the only way this page can be driven
// without a mouse, which is how it was tested.)
// Closing is an IPC call like any other and can be refused: `close` is not in
// `core:window:default`, and a capability lists the windows it covers, so for
// as long as `capabilities/default.json` said `"windows": ["bar"]` this
// rejected and the button did nothing at all. Never call it bare again -- a
// silent `.close()` is indistinguishable from a dead button.
function closeWindow() {
  win.close().catch((e) => say(`關不掉這個視窗：${e}`));
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeWindow();
  else if (e.key.toLowerCase() === "s" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    save();
  }
});

$("save").addEventListener("click", save);

async function save() {
  // Read before the form overwrites it: the bar restarts only when the source
  // really changed, and saying so is the only sign the change was taken.
  const wasSource = cfg.audio.source;
  cfg.ui = readUi();
  cfg.hotkeys = {
    click_through: $("hk_click_through").value.trim(),
    pause: $("hk_pause").value.trim(),
    settings: $("hk_settings").value.trim(),
  };
  cfg.audio.source = $("source").value;
  cfg.transcript.enabled = $("tr_enabled").checked;
  cfg.transcript.bilingual_file = $("tr_bilingual").checked;
  cfg.transcript.dir = $("tr_dir").value.trim();
  cfg.transcript.formats = FORMATS.filter((f) => $(`fmt_${f}`).checked);
  try {
    const result = await invoke("save_config", { cfg });
    saved = true;
    original = { ...cfg.ui };
    problems(result.hotkeys_refused);
    say(
      cfg.audio.source === wasSource
        ? `已寫入 ${result.path}`
        : `已寫入 ${result.path}，正在切換音訊來源（模型會重新載入）…`,
    );
  } catch (e) {
    say(`存檔失敗：${e}`);
  }
}

$("close").addEventListener("click", closeWindow);

// 關閉 closes this window; 結束程式 ends the program. They were one button
// until a packaged build made it obvious that nothing on screen could stop the
// bar at all -- it has no decorations and stays off the taskbar, so the only
// way out was `pkill`. The Rust side stops the engine and waits for the
// transcript to be written before it exits, which is why this is a command and
// not `window.close()` on every window.
$("quit").addEventListener("click", () => {
  say("正在停止並寫完逐字稿…");
  invoke("quit").catch((e) => say(`結束失敗：${e}`));
});

// Closing without saving puts the bar back -- but from the Rust side, on the
// window's `Destroyed` event. Doing it here does not work: measured, the
// window was torn down while the revert was still on its way, and the bar kept
// a size nobody had agreed to. `original` stays for the same reason it is
// read: it is what this page shows, not what the bar is put back to.

// Size the window to the form, in the pixels the window is measured in rather
// than the ones the page is laid out in. Same disagreement as the bar's
// `fit_bar`: WebKitGTK lays out at `devicePixelRatio` 1.6 here while Tauri
// sizes windows at 1.0, so a window built at 560x720 shows two thirds of this.
let lastHeight = 0;
function fit() {
  const height = Math.ceil(document.body.scrollHeight) + 8;
  // Only when it actually changed, and never for a couple of pixels: resizing
  // the window relays out the page, which can measure a pixel differently.
  if (Math.abs(height - lastHeight) < 8) return;
  lastHeight = height;
  invoke("fit_settings", { width: 520, height, dpr: devicePixelRatio })
    .catch((e) => console.error("fit_settings", e));
}
window.addEventListener("resize", fit);

load().catch((e) => say(`讀不到設定：${e}`));
