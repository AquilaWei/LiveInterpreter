//! Where this program's own files live, on each platform (PLAN §14, §15).
//!
//! Three crates needed the user's home directory and each worked it out for
//! itself, which is how task 1.11 found this: two of them had grown a
//! `USERPROFILE` fallback for Windows and the one that reads `config.toml` had
//! not. That failure is quiet in the worst way -- `var("HOME")` on Windows does
//! not fail loudly, it returns `Err` into an `unwrap_or_default()`, so the path
//! becomes `.config/liveinterpreter/config.toml`, *relative to whatever
//! directory the program was started in*. Settings would appear to save and
//! come back empty, or come back different depending on how the program was
//! launched.
//!
//! So the rules live here, once, and the platform is a parameter rather than a
//! `cfg!` buried in an expression -- which means the Windows rules can be
//! tested on the Linux machine this was written on, the only part of task 1.11
//! that can be.
//!
//! The layouts are each platform's own convention rather than one shared
//! layout, because these directories are the user's, not ours: on Windows
//! settings belong in `%APPDATA%` where the backup tools look, and a
//! multi-gigabyte model cache belongs in `%LOCALAPPDATA%`, which does not
//! follow a roaming profile onto another machine.
//!
//! | | Linux/macOS | Windows |
//! |---|---|---|
//! | settings | `$XDG_CONFIG_HOME/liveinterpreter/` or `~/.config/liveinterpreter/` | `%APPDATA%\LiveInterpreter\` |
//! | models | `$XDG_CACHE_HOME/liveinterpreter/models/` or `~/.cache/liveinterpreter/models/` | `%LOCALAPPDATA%\LiveInterpreter\models\` |
//! | transcripts | `~/Documents/LiveInterpreter/` (`[transcript] dir`) | `%USERPROFILE%\Documents\LiveInterpreter\` |

use std::path::PathBuf;

/// Which set of rules to apply. [`Platform::HERE`] is this build's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Unix,
    Windows,
}

impl Platform {
    pub const HERE: Platform = if cfg!(windows) {
        Platform::Windows
    } else {
        Platform::Unix
    };
}

/// The user's home directory.
///
/// `HOME` first even on Windows: it is not normally set there, but when it is
/// -- Git Bash, MSYS, WSL interop, a deliberate override -- it is set by
/// somebody who meant it.
pub fn home() -> Option<PathBuf> {
    home_from(Platform::HERE, env)
}

/// `config.toml`. `$LI_CONFIG` overrides it and is checked by the caller.
pub fn config_file() -> PathBuf {
    config_file_from(Platform::HERE, env)
}

/// The shared model cache. `$LI_MODEL_DIR` overrides it and is checked by the
/// caller.
pub fn model_cache() -> PathBuf {
    model_cache_from(Platform::HERE, env)
}

/// An unset variable and an empty one are the same thing here: a launcher that
/// exports `HOME=` has told us nothing.
fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}

fn home_from(p: Platform, env: impl Fn(&str) -> Option<String>) -> Option<PathBuf> {
    env("HOME")
        .or_else(|| match p {
            Platform::Windows => env("USERPROFILE"),
            Platform::Unix => None,
        })
        .map(PathBuf::from)
}

/// With no home directory to be found, the path stays relative rather than
/// being invented. There is no better guess, and a relative path at least
/// shows up as one in the settings window's "設定檔" line.
fn config_file_from(p: Platform, env: impl Fn(&str) -> Option<String>) -> PathBuf {
    match p {
        Platform::Windows => match env("APPDATA")
            .map(PathBuf::from)
            .or_else(|| home_from(p, &env))
        {
            Some(base) => base.join("LiveInterpreter/config.toml"),
            None => PathBuf::from("LiveInterpreter/config.toml"),
        },
        Platform::Unix => {
            let base = env("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| home_from(p, &env).unwrap_or_default().join(".config"));
            base.join("liveinterpreter/config.toml")
        }
    }
}

fn model_cache_from(p: Platform, env: impl Fn(&str) -> Option<String>) -> PathBuf {
    match p {
        Platform::Windows => env("LOCALAPPDATA")
            .map(PathBuf::from)
            .or_else(|| home_from(p, &env))
            .unwrap_or_default()
            .join("LiveInterpreter/models"),
        // `XDG_CACHE_HOME` before `$HOME/.cache`, the same way the config
        // file honours `XDG_CONFIG_HOME` above. This was written the other way
        // round until task 1.14b, and in a flatpak the difference is a
        // gigabyte: the sandbox gives the app a **tmpfs** home and points
        // `XDG_CACHE_HOME` at the one directory that survives
        // (`~/.var/app/<id>/cache`). Measured -- the downloader fetched all
        // 1003 MB, checked every hash, and the models were gone as soon as the
        // process exited, so every run would have re-downloaded them.
        Platform::Unix => {
            let base = env("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| home_from(p, &env).unwrap_or_default().join(".cache"));
            base.join("liveinterpreter/models")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake environment. Anything not listed is unset -- including `HOME` on
    /// the Windows cases, which is the whole point.
    fn envs<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| (*v).to_owned())
                .filter(|s| !s.is_empty())
        }
    }

    #[test]
    fn linux_settings_go_under_dot_config() {
        let p = config_file_from(Platform::Unix, envs(&[("HOME", "/home/a")]));
        assert_eq!(
            p,
            PathBuf::from("/home/a/.config/liveinterpreter/config.toml")
        );
    }

    #[test]
    fn xdg_config_home_wins_on_linux() {
        let p = config_file_from(
            Platform::Unix,
            envs(&[("HOME", "/home/a"), ("XDG_CONFIG_HOME", "/tmp/cfg")]),
        );
        assert_eq!(p, PathBuf::from("/tmp/cfg/liveinterpreter/config.toml"));
    }

    #[test]
    fn windows_settings_go_to_appdata() {
        // The bug this module exists for: with no HOME set, the old rule made
        // this path relative and the settings window wrote wherever it was
        // started from.
        let p = config_file_from(
            Platform::Windows,
            envs(&[
                ("APPDATA", r"C:\Users\a\AppData\Roaming"),
                ("USERPROFILE", r"C:\Users\a"),
            ]),
        );
        assert_eq!(
            p,
            PathBuf::from(r"C:\Users\a\AppData\Roaming").join("LiveInterpreter/config.toml")
        );
        assert!(p.is_absolute() || p.to_string_lossy().starts_with(r"C:\"));
    }

    #[test]
    fn windows_falls_back_to_the_profile_when_appdata_is_missing() {
        let p = config_file_from(Platform::Windows, envs(&[("USERPROFILE", r"C:\Users\a")]));
        assert_eq!(
            p,
            PathBuf::from(r"C:\Users\a").join("LiveInterpreter/config.toml")
        );
    }

    #[test]
    fn xdg_config_home_is_ignored_on_windows() {
        // Set by MSYS and by some cross-platform tools; following it there
        // would scatter settings into a shell's idea of the filesystem.
        let p = config_file_from(
            Platform::Windows,
            envs(&[("APPDATA", r"C:\Roaming"), ("XDG_CONFIG_HOME", "/tmp/cfg")]),
        );
        assert!(p.starts_with(r"C:\Roaming"));
    }

    #[test]
    fn models_go_to_the_local_appdata_which_does_not_roam() {
        let p = model_cache_from(
            Platform::Windows,
            envs(&[
                ("APPDATA", r"C:\Users\a\AppData\Roaming"),
                ("LOCALAPPDATA", r"C:\Users\a\AppData\Local"),
            ]),
        );
        assert_eq!(
            p,
            PathBuf::from(r"C:\Users\a\AppData\Local").join("LiveInterpreter/models")
        );
    }

    #[test]
    fn models_stay_in_dot_cache_on_linux() {
        let p = model_cache_from(Platform::Unix, envs(&[("HOME", "/home/a")]));
        assert_eq!(p, PathBuf::from("/home/a/.cache/liveinterpreter/models"));
    }

    #[test]
    fn xdg_cache_home_wins_over_the_home_directory() {
        // The flatpak case (task 1.14b), and the reason this is not cosmetic:
        // inside the sandbox `$HOME` is a tmpfs and only the directory named
        // here survives the process. Getting this wrong throws away a
        // gigabyte of downloaded models on every exit.
        let p = model_cache_from(
            Platform::Unix,
            envs(&[
                ("HOME", "/home/a"),
                ("XDG_CACHE_HOME", "/home/a/.var/app/tw.example.App/cache"),
            ]),
        );
        assert_eq!(
            p,
            PathBuf::from("/home/a/.var/app/tw.example.App/cache/liveinterpreter/models")
        );
    }

    #[test]
    fn an_empty_xdg_cache_home_falls_back_like_an_unset_one() {
        // Same rule as everywhere else here: a launcher that exports
        // `XDG_CACHE_HOME=` has told us nothing.
        let p = model_cache_from(
            Platform::Unix,
            envs(&[("HOME", "/home/a"), ("XDG_CACHE_HOME", "")]),
        );
        assert_eq!(p, PathBuf::from("/home/a/.cache/liveinterpreter/models"));
    }

    #[test]
    fn xdg_cache_home_is_ignored_on_windows() {
        // Same shape as the config rule above: the Windows layout is Windows'
        // own, not an XDG one wearing a hat.
        let p = model_cache_from(
            Platform::Windows,
            envs(&[
                ("LOCALAPPDATA", r"C:\Users\a\AppData\Local"),
                ("XDG_CACHE_HOME", "/tmp/cache"),
            ]),
        );
        assert_eq!(
            p,
            PathBuf::from(r"C:\Users\a\AppData\Local").join("LiveInterpreter/models")
        );
    }

    #[test]
    fn an_explicit_home_wins_on_windows_too() {
        // Git Bash and MSYS set it, and somebody who sets it means it.
        let h = home_from(
            Platform::Windows,
            envs(&[("HOME", r"C:\msys\home\a"), ("USERPROFILE", r"C:\Users\a")]),
        );
        assert_eq!(h, Some(PathBuf::from(r"C:\msys\home\a")));
    }

    #[test]
    fn an_empty_variable_counts_as_unset() {
        let h = home_from(
            Platform::Windows,
            envs(&[("HOME", ""), ("USERPROFILE", r"C:\Users\a")]),
        );
        assert_eq!(h, Some(PathBuf::from(r"C:\Users\a")));
    }

    #[test]
    fn nothing_set_leaves_a_relative_path_rather_than_inventing_one() {
        let p = config_file_from(Platform::Unix, envs(&[]));
        assert_eq!(p, PathBuf::from(".config/liveinterpreter/config.toml"));
        assert!(!p.is_absolute());
    }

    #[test]
    fn this_build_resolves_all_three_without_panicking() {
        let _ = home();
        assert!(config_file().ends_with("config.toml"));
        assert!(model_cache().ends_with("models"));
    }
}
