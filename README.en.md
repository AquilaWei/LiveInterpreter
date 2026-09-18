# LiveInterpreter

[![ci](https://github.com/AquilaWei/LiveInterpreter/actions/workflows/ci.yml/badge.svg)](https://github.com/AquilaWei/LiveInterpreter/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[中文](README.md) | **English**

**Turns the English your computer is playing into live Traditional Chinese subtitles at the bottom of the screen.**

In a meeting, a talk or a podcast, a subtitle bar floats at the bottom of the screen: the
English original on top, the Chinese translation below. Speech recognition and translation
both run on your own computer, so **the audio never leaves the machine**. A transcript is
left behind when you finish.

> [!NOTE]
> **This is a vibe-coding project.** Almost all of the code, tests and documentation were
> written by AI (Anthropic's Claude, through Claude Code). The maintainer sets the
> requirements, makes the decisions, and uses and accepts the result.
>
> That means the code **has not been reviewed line by line by a person**. Confidence comes
> from measurement rather than review: 300+ automated tests, CI, and latency and error
> rates measured on real audio. Please use it on that basis, and open an issue if you find
> a problem.

---

## Features

- **Fast and accurate, no trade-off**: a fast lane puts English on screen in about 0.7 s,
  then an accurate lane overwrites the same line with a better version.
- **Fully offline**: recognition and translation run locally; the network is only needed
  to download the models the first time.
- **Uses the GPU when there is one**: the accurate lane runs on Vulkan and is not tied to a
  GPU vendor (so far only tested on Intel Arc). Without a GPU it falls back to the CPU.
- **A subtitle bar that stays out of the way**: always on top, draggable, with switchable
  click-through and global hotkeys.
- **Transcripts**: txt, srt, vtt and jsonl, written as each line is finalised, so a crash
  does not lose what has already been said.
- **Taiwan usage**: translations are converted to the Traditional Chinese used in Taiwan
  with OpenCC `s2twp`.

## How it works

```
Audio being played ──┬─► Fast-lane ASR (CPU) ────────────────► Subtitles (~0.7 s)
                     │                                          ▲ overwritten in place
                     └─► Accurate-lane ASR (GPU) ──► Translate ─┴─► Transcript
```

Both lanes listen to the same audio at the same time. The fast lane is a streaming model
that writes as it hears; the accurate lane waits for the end of a sentence and recognises
it as a whole, replacing the line about a second later. The translation is updated with it.

---

## Status

| Platform | Status |
|---|---|
| **Linux (Flatpak)** | ✅ Recommended. Fully verified on the development machine; installed from nothing in a clean Debian 13 container |
| Linux (rpm / deb) | ✅ Works, but only on systems with glibc ≥ 2.43 (e.g. Fedora 44 and later) |
| Windows | ⏸ On hold. Some code exists; it has never been run on Windows |
| Android | ⏸ On hold |

The only language direction for now is **English → Traditional Chinese**.

Nobody has yet checked the display, audio and GPU on a second physical computer. If you try
it, please open an issue whether it works or not.

---

## Install

### 1. Install the package

Download `LiveInterpreter.flatpak` from [Releases](https://github.com/AquilaWei/LiveInterpreter/releases):

```bash
flatpak install --user ./LiveInterpreter.flatpak
```

The GNOME runtime it needs is fetched from Flathub automatically. A CJK font is bundled.

rpm and deb packages are also provided (glibc ≥ 2.43 required):

```bash
sudo dnf install ./LiveInterpreter-*.x86_64.rpm
sudo apt install ./LiveInterpreter_*_amd64.deb      # not tested
```

### 2. Download the models

The package does not include the models (about 1 GB). On first launch the app offers to
download them and shows the progress; you can also use the command line:

```bash
flatpak run --command=liveinterpreter io.github.AquilaWei.LiveInterpreter --fetch-models
```

Interrupted downloads resume, and every file is checked against its sha256; a mismatch is
deleted and reported. Updating the app does not re-download the models.

---

## Usage

```bash
flatpak run io.github.AquilaWei.LiveInterpreter
```

Or open it from the application menu. By default it captures **the audio your computer is
playing**; to listen to a microphone instead, switch the source in the settings window.

| Hotkey | Action |
|---|---|
| `Ctrl+Alt+C` | Toggle click-through (clicks go to whatever is behind the bar) |
| `Ctrl+Alt+P` | Pause / resume |
| `Ctrl+Alt+S` | Open the settings window |

Hotkeys can be changed under `[hotkeys]` in the config file.

The **command-line version** prints subtitles to the terminal, which is handy for debugging
or for machines without a desktop:

```bash
flatpak run --command=liveinterpreter io.github.AquilaWei.LiveInterpreter                 # start
flatpak run --command=liveinterpreter io.github.AquilaWei.LiveInterpreter --list-devices  # list audio and GPUs
flatpak run --command=liveinterpreter io.github.AquilaWei.LiveInterpreter --help
```

With the rpm / deb packages, the commands are `liveinterpreter-desktop` and `liveinterpreter`.

---

## Configuration

Font size, opacity, number of lines and position are set in the settings window and take
effect immediately. Everything else is in `config.toml`:

| Installed as | Config file |
|---|---|
| Flatpak | `~/.var/app/io.github.AquilaWei.LiveInterpreter/config/liveinterpreter/config.toml` |
| rpm / deb | `~/.config/liveinterpreter/config.toml` |

The settings window shows the actual path at the top. Without a config file the defaults
are used, and the defaults were chosen by measurement. Every option is documented in
[`crates/li-core/src/config.rs`](crates/li-core/src/config.rs).

Transcripts are written to `~/Documents/LiveInterpreter/` by default.

---

## Building from source

```bash
git clone https://github.com/AquilaWei/LiveInterpreter
cd LiveInterpreter
./scripts/build.sh                  # use this script, not a bare cargo build
./scripts/run.sh --list-devices
```

Read [`BUILDING.md`](BUILDING.md) first: the Vulkan backend needs SPIRV-Headers, which some
distributions do not package. The Flatpak is built with `./scripts/flatpak.sh`; the host
does not need flatpak-builder installed.

Tests:

```bash
./scripts/build.sh test --workspace
```

Tests that need the models print `SKIP` and pass when the models are absent.
`cargo xtask eval` re-runs the latency and error-rate measurements on the audio in
[`testdata/`](testdata/README.md).

---

## Tech stack

| Part | Uses |
|---|---|
| Language | Rust (a workspace of 8 crates, plus the CLI and desktop app) |
| Fast-lane ASR | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) streaming Zipformer (CPU), plus a small model that restores punctuation and casing |
| Accurate-lane ASR | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) `small.en` q5_1 (Vulkan); `base.en` without a GPU |
| Voice activity detection | Silero VAD, ONNX Runtime |
| Translation | NLLB-200-distilled-600M int8, [CTranslate2](https://github.com/OpenNMT/CTranslate2) + oneDNN (CPU) |
| Simplified → Traditional | OpenCC `s2twp`, dictionaries compiled into the executable |
| Audio capture | PulseAudio (Linux), cpal |
| UI | [Tauri 2](https://tauri.app/), plain HTML + JavaScript front end |
| Packaging | Flatpak, rpm, deb |
| CI | GitHub Actions: `cargo fmt`, `clippy -D warnings`, all tests |

---

## License

The code is licensed under **Apache-2.0** ([`LICENSE`](LICENSE)). Third-party assets carry
their own licenses; the full list is in [`NOTICE`](NOTICE).

> [!IMPORTANT]
> The default translation model, **NLLB-200-distilled-600M, is CC-BY-NC-4.0 and may not be
> used commercially**. It is neither in the repository nor in the package; it is downloaded
> on first run. Personal use is fine; for commercial use, the translation backend has to be
> replaced (see `crates/li-mt`).

---

## More

- [`CHANGELOG.md`](CHANGELOG.md) — what changed in each version and why (in Chinese)
- [`BUILDING.md`](BUILDING.md) — build details and per-platform pitfalls
- [`testdata/README.md`](testdata/README.md) — where the evaluation audio comes from and what it is for
