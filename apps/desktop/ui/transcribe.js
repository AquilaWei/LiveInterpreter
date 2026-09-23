// The 語音轉檔 window: pick an audio file, pick the languages, get a .txt.
//
// The work is `li_core::batch` on the Rust side. Progress arrives as
// `transcribe://progress` events for the same reason as the model download's:
// a long file takes minutes, and a command answers once. The command's own
// answer is the path written, or null when it was cancelled.

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const win = window.__TAURI__.window.getCurrentWindow();

const $ = (id) => document.getElementById(id);

// The last choice of languages, for the next time. A per-viewer convenience:
// with storage unavailable the window still works and starts on 中英對照.
const KEY = "transcribe.output";

let file = null;
let written = null;
let running = false;

const clock = (secs) => {
  const s = Math.max(0, Math.floor(secs));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
};

function output() {
  return document.querySelector("input[name=output]:checked").value;
}

function remember() {
  try {
    localStorage.setItem(KEY, output());
  } catch {
    // Private storage off: forgetting the choice is all that is lost.
  }
}

function recall() {
  let saved = null;
  try {
    saved = localStorage.getItem(KEY);
  } catch {
    return;
  }
  const radio = document.querySelector(`input[name=output][value="${saved}"]`);
  if (radio) radio.checked = true;
}

function busy(on) {
  running = on;
  $("pick").disabled = on;
  $("go").disabled = on || !file;
  $("cancel").hidden = !on;
  $("cancel").disabled = false;
  for (const r of document.querySelectorAll("input[name=output]")) r.disabled = on;
}

function fail(msg) {
  $("error").textContent = msg;
  $("error").hidden = false;
}

async function pick() {
  const path = await invoke("pick_audio").catch((e) => {
    fail(String(e));
    return null;
  });
  if (!path) return;
  file = path;
  $("name").textContent = path.split("/").pop();
  $("name").title = path;
  $("go").disabled = false;
  $("show").hidden = true;
  $("error").hidden = true;
  $("fill").style.width = "0";
  $("status").textContent = "";
}

async function start() {
  if (!file) return;
  remember();
  busy(true);
  written = null;
  $("error").hidden = true;
  $("show").hidden = true;
  $("fill").style.width = "0";
  // Decoding and loading whisper come before the first sentence, and take
  // seconds with nothing to count.
  $("status").textContent = "讀取音訊、載入模型…";
  try {
    written = await invoke("transcribe_file", { path: file, output: output() });
    if (written) {
      $("fill").style.width = "100%";
      $("status").textContent = `完成：${written}`;
      $("show").hidden = false;
    } else {
      $("fill").style.width = "0";
      $("status").textContent = "已取消，沒有寫入任何檔案。";
    }
  } catch (e) {
    $("status").textContent = "";
    fail(`轉檔失敗：${e}`);
  }
  busy(false);
}

function onProgress(p) {
  if (!running || !(p.total_s > 0)) return;
  $("fill").style.width = `${Math.min(100, (p.done_s / p.total_s) * 100).toFixed(1)}%`;
  // After 取消 the status says it is stopping, and a count would hide that.
  if (!$("cancel").disabled) $("status").textContent = `${clock(p.done_s)} / ${clock(p.total_s)}`;
}

$("pick").addEventListener("click", pick);
$("go").addEventListener("click", start);
$("cancel").addEventListener("click", () => {
  $("cancel").disabled = true;
  $("status").textContent = "正在停止（這一句處理完就停）…";
  invoke("cancel_transcribe").catch((e) => fail(String(e)));
});
$("show").addEventListener("click", () => {
  if (written) invoke("show_in_folder", { path: written }).catch((e) => fail(String(e)));
});
// Closing mid-run cancels, on the Rust side, on the window's `Destroyed`
// event: that also catches the title bar's close button, which never reaches
// this page.
$("close").addEventListener("click", () => win.close());

recall();
listen("transcribe://progress", (ev) => onProgress(ev.payload)).catch((e) =>
  fail(`收不到進度：${e}`),
);
