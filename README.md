# LiveInterpreter

即時口譯字幕工具：擷取聲音 → 串流語音辨識 → 翻譯 → 螢幕下方懸浮字幕（原文一行 + 譯文一行）。

- 平台目標：Linux、Windows、Android
- 語言：英文 → 繁體中文（首版）
- 本機小模型可離線；有網路時可切雲端拚精準
- 翻譯前的原文逐字稿可另存獨立檔案

完整規劃見 [`docs/PLAN.md`](docs/PLAN.md)。

## 目前狀態：Phase 1 — 桌面 MVP 開發中

選型 spike（任務 1.0a–1.0c）已完成，兩條 lane 鎖定；Rust workspace 骨架（任務 1.1）已建立。

| | 定案 | 驗收 |
|---|---|---|
| 快線（螢幕字幕）| sherpa-onnx streaming Zipformer，CPU | **G1 ✅** 延遲中位 0.67–0.76s、max 0.98s |
| 精準線（逐字稿 + 翻譯來源）| whisper.cpp `small.en` q5_1 + Vulkan | **G2a** content WER 15.2 / 11.3%；**G2b** 漏句率靠合流守門 |

```bash
cargo test --workspace      # Rust 骨架
```

`poc/` 是 Python 拋棄式原型，保留作為各 crate 的行為對照與量測 harness。

```bash
cd poc
uv sync
uv run li-poc --source system     # 擷取系統聲音，terminal 顯示雙行字幕
```

詳見 [`poc/README.md`](poc/README.md)。
