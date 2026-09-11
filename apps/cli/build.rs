//! Make the built binary find `libsherpa-onnx-c-api.so` next to itself.
//!
//! sherpa-onnx is a shared library, and `sherpa-rs-sys` copies it (with
//! `libonnxruntime.so`, from `ort`'s `copy-dylibs`) into the target directory
//! beside the executable. `cargo run` works because cargo puts that directory
//! on the loader path for the child process; **anything else does not**, and
//! the binary dies before `main` with
//!
//! ```text
//! error while loading shared libraries: libsherpa-onnx-c-api.so
//! ```
//!
//! PLAN §17 predicted this would block the 1.8 alpha rather than wait for the
//! packaging task, and it did, on the first run of the assembled engine. One
//! rpath entry fixes it for the built binary and for anything installed
//! alongside its libraries.
//!
//! `cargo:rustc-link-arg` is per-package, so a binary crate has to say this for
//! itself -- a library's build script cannot say it on a dependent's behalf.
//! The Tauri app of task 1.9 needs the same three lines.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    stamp();
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target.as_str() {
        "linux" | "android" => {
            // The build tree: the .so files sit beside the executable.
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            // The installed tree: the executable is /usr/bin/<name> and the
            // packages put the libraries in a private directory of their own
            // (1.14; PLAN §19-33). Both spellings, because Fedora and Debian
            // disagree about which of lib/ and lib64/ is the real one.
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib/LiveInterpreter");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib64/LiveInterpreter");
        }
        "macos" | "ios" => println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path"),
        // Windows has no rpath: the loader looks beside the .exe already.
        _ => {}
    }
}

/// Stamp the binary with the commit it was built from.
///
/// `Cargo.toml` answers "which release is this". Until now nothing answered
/// "which *build* is this", and two packages a week apart both calling
/// themselves `0.1.0` cost an afternoon of installing one and testing the
/// other. This goes into `LI_BUILD_ID`, is read back with `env!`, and is
/// logged once at startup so `journalctl --user` has it without being asked.
///
/// Duplicated in the other app's build script, like the rpath block above --
/// but for a different reason: rpath is per-package because cargo says so,
/// this is per-package because `env!` reads only what its own package's script
/// emitted. Keep the two the same shape.
///
/// Outside a git checkout -- a source tarball -- the commit is unknown and the
/// version stands on its own.
fn stamp() {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
            .filter(|s| !s.is_empty())
    };
    let id = match (
        git(&["rev-parse", "--short=8", "HEAD"]),
        git(&["log", "-1", "--format=%cs"]),
    ) {
        (Some(hash), Some(date)) => {
            // Untracked files are not dirt: `testdata/private/` and the build
            // tree live here on purpose and say nothing about the source.
            let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).is_some();
            format!("{hash} {date}{}", if dirty { ", modified" } else { "" })
        }
        _ => "unknown commit".to_owned(),
    };
    println!("cargo:rustc-env=LI_BUILD_ID={id}");
    // Without these the stamp stays at whatever it was when this package last
    // had some other reason to rebuild -- a wrong answer, which is worse than
    // no answer, because this exists to be believed.
    if let Some(dir) = git(&["rev-parse", "--git-dir"]) {
        println!("cargo:rerun-if-changed={dir}/HEAD");
        println!("cargo:rerun-if-changed={dir}/index");
    }
}
