# LiveInterpreter

即時口譯字幕工具：擷取聲音 → 串流語音辨識 → 翻譯 → 螢幕下方懸浮字幕（原文一行 + 譯文一行）。

- 平台目標：Linux、Windows、Android
- 語言：英文 → 繁體中文（首版）
- 本機小模型可離線；有網路時可切雲端拚精準
- 翻譯前的原文逐字稿可另存獨立檔案

完整規劃見 [`docs/PLAN.md`](docs/PLAN.md)。

## 目前狀態：Phase 1 — 在 Linux 上自用穩定（v1.1.0）

懸浮字幕條、雙線辨識、翻譯、設定視窗、逐字稿、rpm/deb 安裝，都在真實音訊上跑過。
**不代表可以公開發佈**：Windows 從沒實際跑過，預設翻譯模型是 CC-BY-NC。
每一版的內容見 [`CHANGELOG.md`](CHANGELOG.md)，任務進度見 `docs/PLAN.md` §17。

| | 定案 | 實測 |
|---|---|---|
| 快線（螢幕字幕）| sherpa-onnx streaming Zipformer int8，CPU | **G1 ✅** 延遲中位 0.67–0.76 s、max 0.98 s |
| 精準線（逐字稿 + 翻譯來源）| whisper.cpp `small.en` q5_1 + Vulkan | **G2a** `ami_meeting` content WER 14.7% ✅／`ami_meeting2` 18.7% ❌ |
| 翻譯 | NLLB-200-distilled-600M int8（CTranslate2）+ OpenCC s2twp | 停在逗號的行從 60–70% 降到 0–20% |

```bash
cargo test --workspace
```

`poc/` 是 Python 拋棄式原型，保留作為各 crate 的行為對照與量測 harness。

```bash
cd poc
uv sync
uv run li-poc --source system     # 擷取系統聲音，terminal 顯示雙行字幕
```

詳見 [`poc/README.md`](poc/README.md)。

## 安裝後第一件事：下載模型

安裝包不含模型（71 MB vs 1 GB），第一次啟動時桌面版會自己問要不要下載。要用命令列：

```bash
liveinterpreter --fetch-models     # 約 1.0 GB，可中斷續傳
```

每個檔案下載後都比對 `assets/models.toml` 裡的 sha256 才就位 —— 那些雜湊是從本專案
實際量測用的那份算出來的，所以下載成功同時也意味著你手上的模型就是 `docs/` 裡那些
數字描述的模型。之後的版本更新不會重抓。

## 授權

程式碼是 **Apache-2.0**（[`LICENSE`](LICENSE)）。但這個 repo 裡不是每一樣東西都適用
它，而程式跑起來用的模型根本不在 repo 裡 —— 完整的一份在 [`NOTICE`](NOTICE)，摘要：

| | 授權 |
|---|---|
| 程式碼、建置腳本、文件 | Apache-2.0 |
| `assets/models/silero_vad.onnx`（唯一進 repo 的模型，2.2 MB） | MIT |
| `testdata/` 的 LibriSpeech 與 AMI 音訊與英文參考稿 | CC-BY-4.0（出處與修改方式見 [`testdata/README.md`](testdata/README.md)） |
| `testdata/*.zh.txt` 中文參考稿 | Claude 產生、**未經人工校對** —— 只能做相對比較，不要引用絕對 chrF |

**預設翻譯模型 NLLB-200-distilled-600M 是 CC-BY-NC-4.0（非商用）。** 它不在 repo、
也不在任何安裝包裡 —— 第一次執行時下載到 `~/.cache/liveinterpreter/models/`。自用沒
問題；要拿去做產品就得換掉，MT 後端在 trait 後面（`crates/li-mt`），選擇記在
`assets/models.toml`。
