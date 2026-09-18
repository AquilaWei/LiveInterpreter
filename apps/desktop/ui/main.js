// The bar's front end.
//
// A file rather than an inline <script>: the CSP is `default-src 'self'`, so
// an inline module is blocked outright. The first probe had it inline
// and reported three Wayland failures that were really this.
//
// ## The one rule
//
// The source row never goes backwards. A line is opened by the fast lane and
// settled by the accurate lane about two seconds later, by which time the
// speaker is usually a sentence further on -- so a settled line often arrives
// after the row has moved on. It is dropped from the screen when that happens.
// It is not lost: `li-transcript` already has it, and the transcript file is
// where the accurate text is meant to end up.
//
// The alternative is what the CLI does -- follow whichever line the
// last event named -- and it makes the row jump back to the previous sentence
// and then forward again, several times a minute. That is the line-jumping
// this bar exists to avoid.
//
// ## Two clocks, two rows
//
// The consequence is that the two rows are not always the same sentence. They
// used to be a whole sentence apart; since the draft translation started early it is
// worked out during the silence that ends the line, so both land at about
// 0.68 s and usually together. The settled pair still arrives later and still
// separately -- 1.14 s for the source, 1.38 s for its translation -- and a
// settled line whose row has moved on is dropped. Holding the source row back
// to match would throw away the fast lane's whole reason for existing: text on screen
// within about a second.
//
// Both rows dim to say the same thing: what you are reading is the fast lane's
// answer and may be replaced. On the source row that is a hypothesis still
// being revised; on the translation row it is the draft, translated from the
// fast lane's sentence during the silence that ends it, which is what puts
// Chinese on the bar 0.8 s earlier than waiting for the accurate lane would.

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const win = window.__TAURI__.window.getCurrentWindow();

const root = document.getElementById("root");
const source = document.getElementById("source");
const target = document.getElementById("target");
const status = document.getElementById("status");

// Settings arrive from `config.toml`; these are only what is on screen for the
// frame before they do.
let dwellMs = 700;

// --- what is on the bar ----------------------------------------------------

const lines = new Map();   // line_id -> { text, settled }
let shown = -1;            // the line the source row is showing
let shownAt = -1e9;        // when it took the row, for the dwell rule
let queued = -1;           // a newer line waiting for the dwell to expire
let timer = null;
let translated = -1;       // the line the translation row is showing

function note(lineId, text, settled) {
  if (lineId < shown) return;          // the one rule
  lines.set(lineId, { text, settled });
  // A line is only ever needed while it is on the bar or about to be; the rest
  // are the transcript's business.
  if (lines.size > 16) lines.delete(lines.keys().next().value);

  if (lineId === shown) {
    paint();                            // in place, under the same id
    return;
  }
  const wait = shownAt + dwellMs - performance.now();
  if (wait <= 0) {
    take(lineId);
  } else {
    // One fast-lane endpoint can close several lines at once, and they would
    // otherwise flash past unread. Whatever is newest when the dwell expires
    // is what gets the row -- being current matters more than being complete,
    // because nothing here is the record of what was said.
    queued = Math.max(queued, lineId);
    if (!timer) timer = setTimeout(release, wait);
  }
}

function release() {
  timer = null;
  if (queued > shown) take(queued);
  queued = -1;
}

function take(lineId) {
  shown = lineId;
  shownAt = performance.now();
  paint();
}

// The fast lane's model carries no case: it emits ALL CAPS, unpunctuated, and
// that is most of what is on the source row most of the time -- the accurate
// lane replaces it about half a second later, properly cased. Capitals are
// slower to read, and this is a row somebody reads while also listening.
//
// The tell is the text itself rather than a flag, because there is no flag that
// means "this came from a model with no case": whichever lane produced it, a
// line with no lowercase letter anywhere was never cased. Proper nouns cannot
// be recovered -- the information is not there -- so this is a guess, and it is
// a guess only on screen. `li-transcript` writes what the engine produced.
function readable(text) {
  if (/[a-z]/.test(text)) return text;
  return text
    .toLowerCase()
    .replace(/^(\W*)(\w)/, (_, lead, first) => lead + first.toUpperCase())
    // The one word English will not forgive. `\b` puts the boundary at the
    // apostrophe too, so "i'm" is covered by the same pass.
    .replace(/\bi\b/g, "I");
}

function paint() {
  const line = lines.get(shown);
  if (!line) return;
  const text = readable(line.text);
  source.classList.toggle("tentative", !line.settled);
  source.textContent = text;
  // While the words are still arriving, the end of the line is the part being
  // read, so an overlong one loses its head rather than its tail. Once it is
  // settled it reads from the start like any other sentence.
  if (!line.settled) keepTail(source, text);
  fit();
}

// Drop words from the front until what is left fits the row budget.
function keepTail(el, text) {
  if (el.scrollHeight <= el.clientHeight) return;
  const words = text.split(" ");
  for (let i = 1; i < words.length && el.scrollHeight > el.clientHeight; i++) {
    el.textContent = "…" + words.slice(i).join(" ");
  }
}

function render(ev) {
  switch (ev.kind) {
    case "partial":
      // A partial can be settled: the fast lane reached its endpoint and will
      // not revise the words again, even though the accurate lane has not
      // answered yet. It reads from the start from then on, like a sentence.
      note(ev.line_id, ev.text, ev.settled);
      break;
    case "final":
      note(ev.line_id, ev.text, true);
      break;
    case "translation":
      // Same rule as the source row, for the same reason -- and it is what
      // keeps a draft from overwriting the settled translation of a line the
      // row has already moved past.
      if (ev.line_id < translated) break;
      translated = ev.line_id;
      target.classList.toggle("tentative", !ev.settled);
      target.textContent = ev.text;
      fit();
      break;
    case "status":
      // "listening" is not news, and a subtitle bar with a permanent label on
      // it is a subtitle bar with less room for subtitles. Everything else --
      // loading, paused, reconnecting, an error -- is worth the corner.
      engineStatus = ev.state === "running" ? "" : ev.text;
      status.classList.toggle("error", ev.state === "error");
      if (!noticeTimer) status.textContent = engineStatus;
      break;
  }
}

listen("engine://event", (e) => render(e.payload));

// --- what the hotkeys just did ---------------------------------------------

// A global hotkey is pressed while looking at something else and gives no sign
// it arrived, so the corner says so for a moment. Without it, a click-through
// toggle that failed to register looks exactly like one that worked.
let engineStatus = "";
let noticeTimer = null;

listen("ui://notice", (e) => {
  status.textContent = e.payload;
  status.classList.remove("error");
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => {
    noticeTimer = null;
    status.textContent = engineStatus;
  }, 2200);
});

// The settings window changed something. Restyle and measure again -- the
// window itself is placed on the Rust side, which already knows.
listen("ui://config", (e) => applyConfig(e.payload));

// --- window geometry -------------------------------------------------------

// The height is the only measurement the Rust side cannot make: it depends on
// the font, on how many rows the translation wrapped to, and on whether the
// checklist is there.
//
// It is reported in CSS pixels together with `devicePixelRatio`, because those
// are not the pixels the window is sized in. See `fit_bar` in `main.rs`.
//
// `setTimeout` and not `requestAnimationFrame`: frame callbacks stop when the
// window is not being drawn -- while it is hidden, minimised, or on another
// virtual desktop -- and the bar would then keep the size it had when it went
// away. Measuring does not need a frame.
let lastHeight = 0;
let fitting = false;

function fit() {
  if (fitting) return;
  fitting = true;
  setTimeout(async () => {
    fitting = false;
    const height = Math.ceil(root.getBoundingClientRect().height);
    // No feedback loop: the width comes from the config, not from the content,
    // so a height change cannot change the wrapping that produced it.
    if (height === lastHeight) return;
    lastHeight = height;
    try {
      // `dpr` goes with it: this height is in CSS pixels, and the window is
      // sized in the screen's. WebKitGTK and Tauri disagree about the ratio --
      // measured 1.6 against 1.0 on this desktop -- so the page has to say
      // which pixels it means.
      await invoke("fit_bar", { height, dpr: devicePixelRatio });
    } catch (e) {
      console.error("fit_bar", e);
    }
  });
}

// The first fit happens at the builder's window width; the real one arrives
// with the first `fit_bar`, and the wrapping can differ.
window.addEventListener("resize", fit);

async function applyConfig(ui) {
  ui = ui || (await invoke("ui_config"));
  const s = document.documentElement.style;
  s.setProperty("--font", `${ui.font_size}px`);
  s.setProperty("--opacity", ui.opacity);
  s.setProperty("--source-rows", ui.source_rows);
  s.setProperty("--target-rows", ui.target_rows);
  source.hidden = !ui.show_source;
  dwellMs = ui.min_dwell_ms;
  fit();
}

// Dragging: press and hold anywhere that is not a control. Where the bar ends
// up is remembered on the Rust side, from the window's own `Moved` event --
// `startDragging` reports nothing back, and the position has to survive the
// next time a translation wraps and the window is re-placed.
document.addEventListener("mousedown", async (e) => {
  if (e.button !== 0 || e.target.closest("button")) return;
  await win.startDragging();
});

// Right-click opens the settings. There is a hotkey for it, but a bar with no
// menu, no title and no icon has to say somewhere what can be done to it, and
// this is the gesture people try on a thing with no controls. (It is not
// reachable while click-through is on -- nothing on the bar is. That is what
// the hotkey is for.)
document.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  invoke("open_settings").catch((err) => console.error("open_settings", err));
});

// --- LI_PROBE=1: the Wayland checklist ------------------------

const params = new URLSearchParams(location.search);
if (params.get("probe") === "1") {
  document.body.classList.add("probe");
  note(1, "And this is our first meeting, surprisingly enough.", true);
  target.textContent = "而這出乎意料地是我們的第一次會議。";
  status.textContent = "probe";
  let on = false;
  // The bar reports on itself. Whether a click "goes through" is a judgement
  // call about another window's reaction; whether this window received the
  // event at all is not. Under XWayland the pointer cannot be measured from
  // outside -- XQueryPointer reports no window even when the pointer is over
  // the bar -- so the sensor has to be in here.
  let mice = 0;
  const counter = document.getElementById("mcount");
  for (const t of ["mousemove", "mousedown", "wheel"]) {
    document.addEventListener(t, () => { counter.textContent = ++mice; }, true);
  }
  const setCt = async (next) => {
    on = next;
    await invoke("set_click_through", { on });
    document.getElementById("ctstate").textContent = on ? "ON" : "off";
    if (on) { mice = 0; counter.textContent = "0"; }
  };
  document.getElementById("ct").addEventListener("click", () => setCt(!on));
  // `LI_PROBE_CT=1` turns it on at load, so click-through can be measured
  // without a mouse -- the toggle is unreachable once it works, and a test
  // that needs a hand cannot be run in CI.
  // `=1` before the window is ever shown, `=delay` well after: the two cannot
  // be told apart by hand, and only the second one resembles pressing the
  // button.
  if (params.get("ct") === "1") setCt(true);
  else if (params.get("ct") === "delay") setTimeout(() => setCt(true), 8000);
}

applyConfig();
