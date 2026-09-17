# Building and running LiveInterpreter

## The short version

```bash
scripts/run.sh --list-devices      # what can be captured
scripts/run.sh                     # subtitle whatever the machine is playing
```

`scripts/run.sh` is `cargo run --release -p li-cli --` with the two things
cargo cannot work out for itself (below). Everything after it goes to the
program; `scripts/build.sh` does the same for any other cargo command, and
`scripts/package.sh` builds the `.rpm` and `.deb` (below).

## What you need

### Fedora / RHEL

```bash
sudo dnf install cmake gcc-c++ ninja-build \
                 pulseaudio-libs-devel \
                 vulkan-headers vulkan-loader-devel glslc
```

**SPIRV-Headers is not packaged by Fedora** and whisper.cpp's Vulkan backend
needs it (task 1.0b). Put it somewhere of its own:

```bash
git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers /tmp/spirv-headers
cmake -S /tmp/spirv-headers -B /tmp/spirv-headers/build \
      -DCMAKE_INSTALL_PREFIX="$HOME/.cache/liveinterpreter/local"
cmake --install /tmp/spirv-headers/build
```

`scripts/build.sh` picks that prefix up. Override it with `LI_PREFIX`.

For the desktop app (task 1.9) also:

```bash
sudo dnf install webkit2gtk4.1-devel javascriptcoregtk4.1-devel libsoup3-devel \
                 gtk3-devel librsvg2-devel libappindicator-gtk3-devel
```

### Windows

Untested: task 1.11 wrote the WASAPI path on a Linux machine and verified it by
cross-compiling, not by running it (`docs/phase1/TASK_1_11_WINDOWS.md` says
exactly what that does and does not prove). Expect to find things.

```powershell
winget install Kitware.CMake Ninja-build.Ninja
```

MSVC (Visual Studio Build Tools, "Desktop development with C++") builds
whisper.cpp and CTranslate2 from source, so it has to be there.

For `gpu-vulkan`, install the **LunarG Vulkan SDK** and let it set `VULKAN_SDK`:
`whisper-rs-sys` panics at build time without that variable, and links
`vulkan-1` out of `%VULKAN_SDK%\Lib`. The SDK ships SPIRV-Headers and glslc, so
the Fedora wart above does not apply here. At run time only the vendor driver's
ICD is needed.

Start with the audio, which is the part 1.11 changed:

```powershell
cargo run -p li-audio --example probe
```

It lists both device lists — inputs, and every output endpoint as a capturable
loopback — then samples one for two seconds and reports frames and peak level.
A non-zero peak with something playing means the loopback path works.

### No GPU, or no SPIRV-Headers

```bash
cargo build --release -p li-cli --no-default-features
```

The accurate lane then runs `base.en` on the CPU, which keeps up but is 4-5 WER
points worse (tasks 1.0b, 1.4). Set `[asr.accurate] model = "base.en-q5_1"` and
`device = "cpu"`.

## Models

**You do not have to fetch these by hand.** The desktop app offers to on its
first run, and the CLI does it on demand:

```
liveinterpreter --fetch-models
```

About 1.0 GB, from Hugging Face and one GitHub release. Each file is checked
against the sha256 in `assets/models.toml` before it is put in place, and an
interrupted transfer resumes from where it stopped. The hashes in that manifest
were computed from the copies this project measured its WER on, so a successful
fetch is also a claim that you have the models the numbers in `docs/` describe.

Everything lands in `$XDG_CACHE_HOME/liveinterpreter/models/`, which is
`~/.cache/liveinterpreter/models/` unless something set that variable — the
Flatpak does, and there it is the difference between keeping the gigabyte and
re-downloading it every run. On Windows it is
`%LOCALAPPDATA%\LiveInterpreter\models\`, which does not follow a roaming
profile. `LI_MODEL_DIR` overrides all of it:

| what | where | size |
|---|---|---|
| fast lane | `sherpa-onnx-streaming-zipformer-en-2023-06-21/` (4 files, int8) | 180 MB |
| accurate lane | `ggml/ggml-small.en-q5_1.bin` | 181 MB |
| translation | `nllb-200-distilled-600m-ct2-int8/` (4 files) | 613 MB |
| punctuation | `sherpa-onnx-online-punct-en-2024-08-06/` | 29 MB |

Only what the config asks for is fetched: a machine with a GPU never downloads
the `base.en-q5_1` fallback, and `[asr.fast] punctuation = false` skips the
punctuation model.

The Silero VAD is compiled into the binary and needs nothing.

**Licences differ between these.** Three are Apache-2.0 and the VAD is MIT, but
the default translator (NLLB) is CC-BY-NC-4.0 — fine to download, use and give
away, not fine to sell. See `NOTICE`.

## Two build-time warts

**`onednn-src` fails to link on lib64 distributions** (Fedora among them): it
hard-codes `-L $OUT_DIR/lib` while oneDNN installs to `$OUT_DIR/lib64`, so the
build stops with `could not find native static library dnnl`. `scripts/build.sh`
creates the symlink and retries; that is all the fix there is until upstream
takes one (PLAN §19-24). oneDNN is worth 2.7x on translation, so it is on by
default — `--no-default-features` drops it along with the GPU.

**The first build takes a while.** whisper.cpp, CTranslate2 and oneDNN are all
built from source. Later builds do not repeat it.

## Packaging (Linux)

```bash
cargo install tauri-cli --version "^2" --locked   # once
scripts/package.sh
```

Output in `target/release/bundle/`: `rpm/LiveInterpreter-<version>-1.x86_64.rpm`
and `deb/LiveInterpreter_<version>_amd64.deb`, ~71 MB each, ~228 MB installed.
Both contain

| path | what |
|---|---|
| `/usr/bin/liveinterpreter-desktop` | the floating bar |
| `/usr/bin/liveinterpreter` | the CLI |
| `/usr/lib/LiveInterpreter/lib{onnxruntime,sherpa-onnx-*}.so` | not optional -- see below |
| `/usr/share/applications/LiveInterpreter.desktop` + hicolor icons | the launcher entry |

**Use `scripts/package.sh`, not `cargo tauri build`.** It builds the CLI first
(the bundler copies that binary out of `target/release`, so it has to be there)
and it goes through `scripts/build.sh` for the SPIRV-Headers prefix and the
oneDNN `lib64` retry.

**Why the three `.so` files are in there.** `ort` and `sherpa-rs-sys` ship
onnxruntime and sherpa-onnx as shared libraries, and both are `DT_NEEDED`
entries rather than anything loaded on demand: without them the binary dies
before `main` with `error while loading shared libraries`. They go in a private
directory and the binaries carry three runpath entries -- `$ORIGIN` for the
build tree, `$ORIGIN/../lib/LiveInterpreter` and `.../lib64/...` for the
installed one (`apps/desktop/src-tauri/build.rs`, `apps/cli/build.rs`).
whisper.cpp, CTranslate2 and oneDNN need none of this: they are static.

**The packages carry no models.** They are ~71 MB against ~1 GB of models, so
the first run downloads them into `~/.cache/liveinterpreter/models/` instead;
see "Models" above. An app update therefore never re-downloads them.

**Untested on Debian/Ubuntu.** The `.deb` builds and its layout matches the
`.rpm`, but the Debian package names in `bundle.linux.deb.depends` have not
been resolved against a real apt. The Fedora names have.

**These two packages only install on glibc >= 2.43.** The binaries carry a
`GLIBC_2.43` requirement (`readelf -V`), which is a property of the machine
they were built on, not of the packaging. Fedora 43 is 2.43; anything older
cannot reach `main`. That is what the Flatpak below is for.

## Packaging (Flatpak)

```bash
scripts/flatpak.sh              # build, install --user, write the bundle
scripts/flatpak.sh --no-bundle  # build and install only
```

`flatpak-builder` does not have to be installed: the script runs
`org.flatpak.Builder`, which is itself a flatpak. The first run downloads about
1.5 GB of SDK:

| | |
|---|---|
| `org.gnome.Platform` / `org.gnome.Sdk` `//50` | the runtime and its SDK |
| `org.freedesktop.Sdk.Extension.rust-stable//25.08` | Rust 1.98.1 |
| `org.freedesktop.Sdk.Extension.llvm21//25.08` | libclang, for bindgen |

The manifest is `packaging/flatpak/io.github.AquilaWei.LiveInterpreter.yml`.
Seven things about it are worth knowing before changing it:

* **It builds the binaries in the SDK rather than unpacking the `.deb`.**
  Tauri's official flatpak recipe does the latter and it does not work here:
  the host `.deb` wants `GLIBC_2.43` and the runtime has 2.42.
* **The build is given the network** (`build-args: [--share=network]`), because
  `ort` and `sherpa-rs-sys` fetch prebuilt libraries in their build scripts and
  cargo fetches the registry. This is what Flathub would not accept; going
  offline means a generated `cargo-sources.json` plus `ORT_LIB_PATH` and
  `SHERPA_LIB_PATH`, and needs no code change.
* **`--device=dri` is not optional**, and `--filesystem=xdg-documents` is the
  one whose absence hurts last: transcripts are written at the end of a
  session, so it would fail after an hour of correct work.
* **The metainfo repeats the version, and `flatpak list` reads that copy**
  rather than the binary -- 1.2.2 shipped calling itself 1.2.1 that way.
  `scripts/flatpak.sh` now refuses to build when the newest `<release>` and
  the workspace `Cargo.toml` disagree, so a release cannot drift again.
* **The bundle carries a `--runtime-repo` pointer**, not the runtime. Without
  it a machine that has never installed `org.gnome.Platform//50` stops at
  "No remote refs found"; with it, `flatpak install ./LiveInterpreter.flatpak`
  adds the remote and pulls the dependency itself.
* **A CJK font is in the package** (`/app/share/fonts`, Noto Sans TC subset,
  5.4 MB, OFL-1.1) because the runtime ships none. Borrowing the host's
  through `/run/host/fonts` works here and produces a row of empty boxes on a
  host that has no CJK font of its own.
* **It goes through `scripts/build.sh` too**, for the SPIRV-Headers prefix and
  the oneDNN `lib64` retry -- the same two warts as everywhere else. `glslc` is
  already in the SDK; SPIRV-Headers is a module of its own.

Models land in `~/.var/app/io.github.AquilaWei.LiveInterpreter/cache/liveinterpreter/models/`.
The sandbox's `$HOME` is a tmpfs, so a build that ignored `XDG_CACHE_HOME`
would throw the gigabyte away on every exit; `li_types::paths` reads it.

There is no auto-update: PLAN §17 1.16 is a version check that only *tells*
you, because Tauri's in-place updater supports AppImage only and the packages
here are rpm/deb.

## Configuration

`~/.config/liveinterpreter/config.toml` (`%APPDATA%\LiveInterpreter\config.toml`
on Windows), or `--config FILE`. There need not be one: the defaults are the
settings tasks 1.4 to 1.7 measured. PLAN §14 is the full schema. The command-line flags override the file for one run and never
write it back.

## Running the tests

```bash
scripts/build.sh test --workspace
scripts/build.sh clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

Some tests need the models and are skipped without them. `cargo xtask eval
--wav testdata/ami_meeting.wav` runs the acceptance harness of PLAN §16, and
`--transcript DIR` makes it write real transcript files while it does.
