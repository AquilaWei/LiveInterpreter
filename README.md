# LiveInterpreter

[![ci](https://github.com/AquilaWei/LiveInterpreter/actions/workflows/ci.yml/badge.svg)](https://github.com/AquilaWei/LiveInterpreter/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**中文** | [English](README.en.md)

**把電腦正在播的英文，即時變成螢幕下方的繁體中文字幕。**

開會、看演講、聽 podcast 時，畫面最下面浮著一條字幕：上面是英文原文，下面是中文翻譯。
語音辨識與翻譯全部在自己的電腦上跑，**音訊不會離開這台機器**。結束後留下一份逐字稿。

> [!NOTE]
> **這是一個 vibe-coding 專案。** 程式碼、測試與文件幾乎全部由 AI（Anthropic 的
> Claude，透過 Claude Code）撰寫；維護者負責提出需求、做決定、實際使用與驗收。
>
> 這代表：程式碼**沒有經過人逐行審查**。可信度來自量測而不是審閱 —— 300 多個自動化
> 測試、CI、以及在真實音訊上量出來的延遲與錯誤率。請以這個前提使用，發現問題歡迎開
> issue。

---

## 特色

- **快又準，不必二選一**：快線約 0.7 秒就把英文打上螢幕，精準線隨後原地覆蓋成更準的版本。
- **完全離線**：辨識與翻譯都在本機；只有第一次下載模型需要網路。
- **有 GPU 就用 GPU**：精準線走 Vulkan，不綁定顯示卡廠牌（目前只在 Intel Arc 上實測）；
  沒有 GPU 自動退回 CPU。
- **不擋路的字幕條**：永遠在最上層、可拖曳、可切換點擊穿透，全域快捷鍵控制。
- **逐字稿**：txt、srt、vtt、jsonl，每句定稿就寫入，程式當掉也不會丟失已經說過的內容。
- **台灣用詞**：翻譯後經 OpenCC `s2twp` 轉成台灣慣用的繁體中文。

## 運作方式

```
電腦在播的聲音 ──┬─► 快線辨識（CPU） ──────────────► 螢幕字幕（約 0.7 秒）
                 │                                        ▲ 原地覆蓋
                 └─► 精準線辨識（GPU） ──► 翻譯 ──────────┴─► 逐字稿
```

兩條辨識線同時跑同一段聲音。快線是串流模型，邊聽邊出字；精準線等一句話說完再整句辨識，
約一秒後把同一行換成更準的文字，翻譯也跟著更新。

---

## 目前狀態

| 平台 | 狀態 |
|---|---|
| **Linux（Flatpak）** | ✅ 建議的安裝方式。開發機上完整驗證；乾淨的 Debian 13 容器中從零安裝成功 |
| Linux（rpm / deb） | ✅ 可用，但只能裝在 glibc ≥ 2.43 的系統（例如 Fedora 44 以後） |
| Windows | ⏸ 暫緩。部分程式碼已寫，從未在 Windows 上執行過 |
| Android | ⏸ 暫緩 |

語言方向目前只有**英文 → 繁體中文**。

還沒有人在「另一台實體電腦」上確認過畫面、聲音與 GPU —— 如果你試了，不論成功與否都歡迎
開 issue 告訴我們。

---

## 安裝

### 1. 安裝套件

到 [Releases](https://github.com/AquilaWei/LiveInterpreter/releases) 下載 `LiveInterpreter.flatpak`：

```bash
flatpak install --user ./LiveInterpreter.flatpak
```

需要的 GNOME runtime 會自動從 Flathub 下載，中文字型已內附。

也提供 rpm / deb（需要 glibc ≥ 2.43）：

```bash
sudo dnf install ./LiveInterpreter-*.x86_64.rpm
sudo apt install ./LiveInterpreter_*_amd64.deb      # 未實測
```

### 2. 下載模型

安裝包不含模型（約 1 GB）。第一次開啟時會自動詢問並顯示下載進度；也可以用命令列：

```bash
flatpak run --command=liveinterpreter io.github.AquilaWei.LiveInterpreter --fetch-models
```

中斷可以續傳，每個檔案都會比對 sha256，不符就刪掉並報錯。更新程式不需要重新下載。

---

## 使用

```bash
flatpak run io.github.AquilaWei.LiveInterpreter
```

或從應用程式選單開啟。預設擷取**電腦正在播的聲音**；要改聽麥克風，到設定視窗切換。

| 快捷鍵 | 作用 |
|---|---|
| `Ctrl+Alt+C` | 切換點擊穿透（滑鼠可以點到字幕條後面的東西） |
| `Ctrl+Alt+P` | 暫停 / 繼續 |
| `Ctrl+Alt+S` | 開啟設定視窗 |

快捷鍵可以在設定檔的 `[hotkeys]` 修改。

**命令列版**會把字幕印在終端機上，適合除錯或沒有桌面的環境：

```bash
flatpak run --command=liveinterpreter io.github.AquilaWei.LiveInterpreter                 # 開始
flatpak run --command=liveinterpreter io.github.AquilaWei.LiveInterpreter --list-devices  # 列出音訊與 GPU
flatpak run --command=liveinterpreter io.github.AquilaWei.LiveInterpreter --help
```

用 rpm / deb 安裝的話，指令是 `liveinterpreter-desktop` 與 `liveinterpreter`。

---

## 設定

字級、透明度、行數、位置在設定視窗裡調，立即生效。完整設定在 `config.toml`：

| 安裝方式 | 設定檔位置 |
|---|---|
| Flatpak | `~/.var/app/io.github.AquilaWei.LiveInterpreter/config/liveinterpreter/config.toml` |
| rpm / deb | `~/.config/liveinterpreter/config.toml` |

設定視窗上方會顯示實際路徑。沒有設定檔時使用預設值，預設值都是量測後選的。每個選項的
說明在 [`crates/li-core/src/config.rs`](crates/li-core/src/config.rs)。

逐字稿預設寫到 `~/Documents/LiveInterpreter/`。

---

## 從原始碼建置

```bash
git clone https://github.com/AquilaWei/LiveInterpreter
cd LiveInterpreter
./scripts/build.sh                  # 請用這個腳本，不要直接 cargo build
./scripts/run.sh --list-devices
```

建置前請先看 [`BUILDING.md`](BUILDING.md)：Vulkan 後端需要 SPIRV-Headers，有些發行版沒有
打包，要自己裝一份。Flatpak 用 `./scripts/flatpak.sh` 建，主機不需要安裝 flatpak-builder。

測試：

```bash
./scripts/build.sh test --workspace
```

需要模型的測試在沒有模型時會印出 `SKIP` 並通過。`cargo xtask eval` 可以在
[`testdata/`](testdata/README.md) 的音檔上重跑延遲與錯誤率量測。

---

## 技術棧

| 部分 | 使用 |
|---|---|
| 語言 | Rust（8 個 crate 的 workspace + 命令列 + 桌面版） |
| 快線辨識 | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) streaming Zipformer（CPU），另有小模型補標點與大小寫 |
| 精準線辨識 | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) `small.en` q5_1（Vulkan），無 GPU 時退回 `base.en` |
| 語音偵測 | Silero VAD，ONNX Runtime |
| 翻譯 | NLLB-200-distilled-600M int8，[CTranslate2](https://github.com/OpenNMT/CTranslate2) + oneDNN（CPU） |
| 簡轉繁 | OpenCC `s2twp`，字典編進執行檔 |
| 音訊擷取 | PulseAudio（Linux）、cpal |
| 介面 | [Tauri 2](https://tauri.app/)，前端為純 HTML + JavaScript |
| 打包 | Flatpak、rpm、deb |
| CI | GitHub Actions：`cargo fmt`、`clippy -D warnings`、全部測試 |

---

## 授權

程式碼採用 **Apache-2.0**（[`LICENSE`](LICENSE)）。第三方素材各有授權，完整清單見
[`NOTICE`](NOTICE)。

> [!IMPORTANT]
> 預設的翻譯模型 **NLLB-200-distilled-600M 是 CC-BY-NC-4.0，不可商用**。它不在 repo 也
> 不在安裝包裡，是第一次執行時下載的。個人使用沒有問題；若要用於商業用途，需要換掉翻譯
> 後端（見 `crates/li-mt`）。

---

## 更多

- [`CHANGELOG.md`](CHANGELOG.md) — 每個版本改了什麼、為什麼
- [`BUILDING.md`](BUILDING.md) — 建置細節與各平台的坑
- [`testdata/README.md`](testdata/README.md) — 評估用音檔的來源與用途
