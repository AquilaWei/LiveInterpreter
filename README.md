# LiveInterpreter

**把電腦正在播的英文，即時變成螢幕下方的中文字幕。**

開會、看演講、聽 podcast 時，畫面最下面浮一條字幕：一行英文原文、一行繁體中文。
全部在自己電腦上跑，**不連網、音訊不離開這台機器**。結束後留下一份逐字稿。

```
電腦在播的聲音 ──┬─► 快線 ASR ──► 螢幕字幕（~0.7 秒）
                 │      （先出，可能有錯）
                 └─► 精準線 ASR ──► 覆蓋修正 ──► 翻譯 ──► 逐字稿存檔
```

兩條辨識線同時跑：快的那條 0.7 秒就把字打上去，慢的那條約 1.2 秒後**原地覆蓋**成
更準的版本。所以你不必在「快」和「準」之間選。

---

## 現在能用到什麼程度

**Linux 桌面，自用穩定。** 懸浮字幕條、設定視窗、逐字稿、安裝包，都在真實音訊上
跑過一段時間了。

**Flatpak 是建議的安裝方式**，而且理由不只是方便：`.rpm` / `.deb` 裡的執行檔要求
glibc ≥ 2.43，實際上只裝得起來在 Fedora 44 以後。Flatpak 自帶 runtime，沒有這個限制。

Flatpak 在開發機（Fedora 44 + Intel Arc 140V）上完整跑過，也在一個乾淨的 Debian 13
容器裡從零安裝成功（沒有 flatpak remote、沒有 runtime、沒有中文字型）。**還沒有人在
另一台真的電腦上看過畫面、聽過聲音、用過 GPU** —— 如果你試了，不管成不成功都歡迎開
issue。

Windows 與 Android 的程式碼寫了一部分，但**都暫緩**，現在不要期待它們能用。

英文 → 繁體中文，單一方向。

---

## 技術棧

| 部分 | 用的是 |
|---|---|
| 語言 | Rust（workspace，8 個 crate + CLI + 桌面版） |
| 快線辨識 | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) streaming Zipformer，CPU；另一個小模型補標點與大小寫 |
| 精準線辨識 | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) `small.en` q5_1，Vulkan GPU（沒有 GPU 時退到 CPU 的 `base.en`） |
| 語音偵測 | Silero VAD，跑在 ONNX Runtime 上 |
| 翻譯 | NLLB-200-distilled-600M int8，跑在 [CTranslate2](https://github.com/OpenNMT/CTranslate2) + oneDNN，CPU |
| 簡轉繁 | OpenCC `s2twp`（台灣用詞），字典編進執行檔 |
| 擷取音訊 | PulseAudio（Linux）／cpal |
| 介面 | [Tauri 2](https://tauri.app/)，前端是純 HTML + JS，沒有 bundler |
| 打包 | Flatpak（主要）、rpm、deb |
| CI | GitHub Actions：`cargo fmt`、`clippy -D warnings`、全部測試 |

---

## 安裝

### 1. 裝套件

到 [Releases](https://github.com/AquilaWei/LiveInterpreter/releases) 下載，然後：

```bash
flatpak install --user ./LiveInterpreter.flatpak    # 建議
```

還是提供 `.rpm` / `.deb`，但**它們需要 glibc ≥ 2.43**：

```bash
sudo dnf install ./LiveInterpreter-*.x86_64.rpm     # Fedora 44 起
sudo apt install ./LiveInterpreter_*_amd64.deb      # Debian/Ubuntu（未實測）
```

### 2. 下載模型

安裝包裡**不含模型**（Flatpak 約 44 MB，模型 1 GB）。第一次打開桌面版它會自己問要不要下載；
要用命令列：

```bash
flatpak run --command=liveinterpreter io.github.AquilaWei.LiveInterpreter --fetch-models
```

約 1.0 GB，中斷可續傳，每個檔案下載後都會比對 sha256 才就位，不符就刪掉並報錯。
之後更新程式不會重抓。（用 rpm/deb 裝的話，指令是 `liveinterpreter --fetch-models`。）

### 3. 跑

```bash
flatpak run io.github.AquilaWei.LiveInterpreter     # 懸浮字幕條（平常用這個）
```

在應用程式選單裡也找得到它。命令列版與裝置列表：

```bash
flatpak run --command=liveinterpreter io.github.AquilaWei.LiveInterpreter
flatpak run --command=liveinterpreter io.github.AquilaWei.LiveInterpreter --list-devices
```

用 rpm/deb 裝的話就是 `liveinterpreter-desktop`、`liveinterpreter`、
`liveinterpreter --list-devices`。

預設擷取**電腦正在播的聲音**。要改聽麥克風：`--source mic`，或在設定視窗裡改。

---

## 從原始碼建

```bash
git clone https://github.com/AquilaWei/LiveInterpreter
cd LiveInterpreter
./scripts/build.sh                    # 不要直接用 cargo build，理由見 BUILDING.md
./scripts/run.sh --list-devices
```

**一定要看 [`BUILDING.md`](BUILDING.md)**：whisper.cpp 的 Vulkan 後端需要
SPIRV-Headers，而 Fedora 沒有打包它，要自己裝一份——`scripts/build.sh` 存在的原因
就是這個。沒有 GPU 也能跑，會自動退到 CPU 的小模型。

---

## 設定

字級、透明度、位置在懸浮條的設定視窗裡改，立刻生效。其餘全部在
`~/.config/liveinterpreter/config.toml`（Flatpak 裝的話是
`~/.var/app/io.github.AquilaWei.LiveInterpreter/config/liveinterpreter/config.toml`；
設定視窗上方就寫著實際路徑）。

逐字稿預設寫到 `~/Documents/LiveInterpreter/`。

---

## 授權

程式碼是 **Apache-2.0**（[`LICENSE`](LICENSE)）。repo 裡不是每樣東西都適用它，
完整的一份在 [`NOTICE`](NOTICE)。

**要注意的一件事**：預設的翻譯模型 NLLB-200-distilled-600M 是 **CC-BY-NC-4.0，
不可商用**。它不在 repo 也不在安裝包裡，是第一次執行時下載的。自己用沒問題；
要拿去做產品就得換掉，MT 後端在 trait 後面（`crates/li-mt`）。

---

## 想看細節

| | |
|---|---|
| [`CHANGELOG.md`](CHANGELOG.md) | 每一版改了什麼、為什麼 |
| [`BUILDING.md`](BUILDING.md) | 建置的細節與各平台的坑 |
| [`testdata/README.md`](testdata/README.md) | 評估用的音檔從哪來、各自量什麼 |

這個專案的規矩是**先量再做**：每個預設值背後都有一次量測，理由寫在設定它的程式碼
註解裡。`cargo xtask eval` 可以在測試音檔上自己重跑延遲與 WER。
