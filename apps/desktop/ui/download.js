// The first-run model download.
//
// This window opens instead of the engine starting, because the engine cannot
// start: `li_core::models::resolve` fails on the first model it cannot find,
// and before this existed that was the end of the program. So the page has one
// job and has to do it without any of the rest of the app being alive.
//
// Progress arrives as events rather than as the command's return value: a
// gigabyte takes minutes, and a command returns once. The Rust side sends one
// `models://progress` per chunk, which is far more often than a screen can be
// read, so the bar is written from a rAF rather than from the event.

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const win = window.__TAURI__.window.getCurrentWindow();

const $ = (id) => document.getElementById(id);

const mib = (n) => {
  const m = n / 1048576;
  return m >= 1024 ? `${(m / 1024).toFixed(2)} GB` : `${Math.round(m)} MB`;
};

let total = 0;
// Latest values from the event stream; the frame loop is what draws them.
let pending = null;
let drawing = false;

function draw() {
  drawing = false;
  if (!pending) return;
  const { stage, name, done } = pending;
  if (done !== null && done !== undefined && total > 0) {
    $("fill").classList.remove("busy");
    $("fill").style.width = `${Math.min(100, (done / total) * 100).toFixed(2)}%`;
    $("bytes").textContent = `${mib(done)} / ${mib(total)}`;
  }
  // Hashing and unpacking move no bytes and can take several seconds on the
  // 613 MB file; without saying so the window looks stuck.
  if (stage === "checking" || stage === "unpacking") {
    $("fill").classList.add("busy");
  } else {
    $("fill").classList.remove("busy");
  }
  const label = { fetching: "下載中", checking: "驗證中", unpacking: "解壓縮", done: "完成" };
  $("stage").textContent = name ? `${label[stage] ?? stage}　${name}` : label[stage] ?? stage;
}

function onProgress(p) {
  if (p.stage === "fetching" && (p.done === null || p.done === undefined) && !p.name) return;
  // Keep the name across the byte-only events, which carry none.
  pending = { ...pending, ...p, name: p.name || (p.done != null ? pending?.name : p.name) };
  if (!drawing) {
    drawing = true;
    requestAnimationFrame(draw);
  }
}

function fail(msg) {
  $("error").textContent = msg;
  $("error").hidden = false;
  $("go").disabled = false;
  $("go").textContent = "重試";
  $("fill").classList.remove("busy");
}

async function start() {
  $("error").hidden = true;
  $("go").disabled = true;
  $("go").textContent = "下載中…";
  $("stage").textContent = "準備…";
  try {
    // Returns when everything is on disk and the engine has started; the Rust
    // side closes this window itself, so there is nothing to do on success.
    await invoke("fetch_models");
  } catch (e) {
    fail(String(e));
  }
}

async function init() {
  await listen("models://progress", (ev) => onProgress(ev.payload));
  let plan;
  try {
    plan = await invoke("model_plan");
  } catch (e) {
    $("intro").textContent = "沒辦法確認要下載什麼。";
    fail(String(e));
    return;
  }
  total = plan.total_bytes;
  $("dir").textContent = plan.dir;
  if (plan.models.length === 0) {
    // Someone put the files there by hand while this window was open. Nothing
    // to do, and saying "0 MB" would be a lie about why it is closing.
    $("intro").textContent = "模型都在了，正在啟動…";
    $("go").hidden = true;
    await invoke("fetch_models").catch((e) => fail(String(e)));
    return;
  }
  $("intro").innerHTML =
    `第一次啟動要先把辨識與翻譯模型抓下來，共 <b>${mib(total)}</b>。` +
    `之後啟動就不用再等。`;
  $("models").replaceChildren(
    ...plan.models.map((m) => {
      const li = document.createElement("li");
      li.textContent = m;
      return li;
    }),
  );
  $("bytes").textContent = `0 MB / ${mib(total)}`;
}

$("go").addEventListener("click", start);
// Hiding rather than quitting: the bar is already up, and someone who wants
// out has the tray and the hotkey. Closing this window does not start an
// engine that has no models, so there is nothing half-started to clean up.
$("hide").addEventListener("click", () => win.close());

init();
