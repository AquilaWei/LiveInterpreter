//! Where a model id in `config.toml` turns into a path on disk (PLAN §15).
//!
//! The config file names models by id (`small.en-q5_1`), the engines want a
//! path, and until now every caller invented its own answer: `xtask eval`
//! hard-coded two paths and `li_mt` hard-coded a third. One place instead, so
//! the downloader of task 1.13 has one convention to write into and the error
//! message when a file is missing says the same thing everywhere.
//!
//! Everything lives under `~/.cache/liveinterpreter/models/`, overridable with
//! `LI_MODEL_DIR` (which is how the tests get a directory of their own):
//!
//! | kind | id | on disk |
//! |---|---|---|
//! | fast lane | `streaming-zipformer-en-2023-06-21` | `sherpa-onnx-<id>/` (a directory of `.onnx` files) |
//! | punctuation | `online-punct-en-2024-08-06` | `sherpa-onnx-<id>/` (the same convention: it is a sherpa release too) |
//! | accurate lane | `small.en-q5_1` | `ggml/ggml-<id>.bin` |
//! | MT | `nllb-200-distilled-600m-ct2-int8` | `<id>/` |
//!
//! An id that is already a path -- absolute, or containing a separator -- is
//! taken as one and left alone, so a model outside the cache needs no
//! ceremony.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

/// Overrides the cache root. Set by the tests; also the escape hatch for a
/// machine that keeps its models on another disk.
pub const DIR_ENV: &str = "LI_MODEL_DIR";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Fast,
    /// The fast lane's punctuation and casing (task 1.25). Optional in a way
    /// the other three are not: a missing one is logged and the engine runs
    /// without it, so `resolve` failing here is not a startup error.
    Punct,
    Accurate,
    Mt,
}

impl Kind {
    fn what(self) -> &'static str {
        match self {
            Kind::Fast => "fast-lane",
            Kind::Punct => "punctuation",
            Kind::Accurate => "accurate-lane",
            Kind::Mt => "translation",
        }
    }

    /// The names to try under the cache root, in order.
    fn candidates(self, id: &str) -> Vec<String> {
        match self {
            // sherpa releases unpack to `sherpa-onnx-<name>`; accepting the
            // bare name too means the config can say either. The punctuation
            // model comes from the same project and unpacks the same way, so
            // it is the same arm rather than a copy of it.
            Kind::Fast | Kind::Punct => vec![format!("sherpa-onnx-{id}"), id.to_owned()],
            Kind::Accurate => vec![
                format!("ggml/ggml-{id}.bin"),
                format!("ggml/{id}"),
                id.to_owned(),
            ],
            Kind::Mt => vec![id.to_owned()],
        }
    }

    fn wants_dir(self) -> bool {
        !matches!(self, Kind::Accurate)
    }
}

/// `$LI_MODEL_DIR`, else the platform's own cache directory
/// (`li_types::paths`).
pub fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(DIR_ENV).filter(|s| !s.is_empty()) {
        return PathBuf::from(dir);
    }
    li_types::paths::model_cache()
}

/// A model cache rooted at one directory.
///
/// A struct rather than a free function so the tests can point at a scratch
/// directory without mutating the process environment -- `set_var` is `unsafe`
/// in edition 2024, and this workspace denies `unsafe_code`.
#[derive(Debug, Clone)]
pub struct Models {
    root: PathBuf,
}

impl Default for Models {
    fn default() -> Self {
        Self::new()
    }
}

impl Models {
    pub fn new() -> Self {
        Self { root: cache_dir() }
    }

    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Turn a config id into a path that exists, or say precisely what is
    /// missing.
    ///
    /// The error names every path that was tried and points at the manifest,
    /// because the alternative -- "No such file or directory" from somewhere
    /// three layers down -- is how an afternoon disappears. Task 1.13's
    /// downloader will turn this into an offer to fetch it; until then it is
    /// instructions.
    pub fn resolve(&self, kind: Kind, id: &str) -> Result<PathBuf> {
        let raw = Path::new(id);
        if raw.is_absolute() || id.contains(std::path::MAIN_SEPARATOR) || id.contains('/') {
            if exists(raw, kind) {
                return Ok(raw.to_path_buf());
            }
            bail!("{} model not found: {}", kind.what(), raw.display());
        }

        let tried: Vec<PathBuf> = kind
            .candidates(id)
            .iter()
            .map(|c| self.root.join(c))
            .collect();
        if let Some(hit) = tried.iter().find(|p| exists(p, kind)) {
            return Ok(hit.clone());
        }

        let list = tried
            .iter()
            .map(|p| format!("\n    {}", p.display()))
            .collect::<String>();
        bail!(
            "{} model {id:?} is not in the model cache.\n  Looked for:{list}\n  \
             See `assets/models.toml` for what to download and where it goes \
             (PLAN §15; the downloader is task 1.13).",
            kind.what()
        )
    }
}

fn exists(p: &Path, kind: Kind) -> bool {
    if kind.wants_dir() {
        p.is_dir()
    } else {
        p.is_file()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> Models {
        let dir = std::env::temp_dir().join("li-core-models").join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Models::with_root(dir)
    }

    #[test]
    fn a_sherpa_release_is_found_by_its_bare_name() {
        let m = scratch("sherpa");
        let want = m
            .root()
            .join("sherpa-onnx-streaming-zipformer-en-2023-06-21");
        std::fs::create_dir_all(&want).unwrap();
        assert_eq!(
            m.resolve(Kind::Fast, "streaming-zipformer-en-2023-06-21")
                .unwrap(),
            want
        );
        // ...and by the name the directory actually has.
        assert_eq!(
            m.resolve(Kind::Fast, "sherpa-onnx-streaming-zipformer-en-2023-06-21")
                .unwrap(),
            want
        );
    }

    #[test]
    fn the_punctuation_model_follows_the_fast_lane_convention() {
        // Same release layout, so the same arm -- and this is the test that
        // notices if someone splits them and forgets the `sherpa-onnx-` prefix.
        let m = scratch("punct");
        let want = m.root().join("sherpa-onnx-online-punct-en-2024-08-06");
        std::fs::create_dir_all(&want).unwrap();
        assert_eq!(
            m.resolve(Kind::Punct, "online-punct-en-2024-08-06")
                .unwrap(),
            want
        );
    }

    #[test]
    fn a_ggml_model_is_a_file_under_ggml() {
        let m = scratch("ggml");
        std::fs::create_dir_all(m.root().join("ggml")).unwrap();
        let want = m.root().join("ggml/ggml-small.en-q5_1.bin");
        std::fs::write(&want, b"x").unwrap();
        assert_eq!(m.resolve(Kind::Accurate, "small.en-q5_1").unwrap(), want);
    }

    #[test]
    fn a_directory_is_not_accepted_where_a_file_is_wanted() {
        // The failure this prevents: a half-finished download left as a
        // directory, and whisper.cpp reporting something unrelated three
        // layers down.
        let m = scratch("wrong_kind");
        std::fs::create_dir_all(m.root().join("ggml/ggml-small.en-q5_1.bin")).unwrap();
        assert!(m.resolve(Kind::Accurate, "small.en-q5_1").is_err());
    }

    #[test]
    fn a_missing_model_names_every_path_it_tried() {
        let m = scratch("missing");
        let err = format!("{:#}", m.resolve(Kind::Mt, "nllb-x").unwrap_err());
        assert!(err.contains("nllb-x"), "{err}");
        assert!(err.contains("models.toml"), "{err}");
    }

    #[test]
    fn an_id_that_is_already_a_path_is_left_alone() {
        let m = scratch("literal");
        let outside = m.root().join("elsewhere/model.bin");
        std::fs::create_dir_all(outside.parent().unwrap()).unwrap();
        std::fs::write(&outside, b"x").unwrap();
        assert_eq!(
            m.resolve(Kind::Accurate, outside.to_str().unwrap())
                .unwrap(),
            outside
        );
    }

    #[test]
    fn the_default_root_is_the_shared_model_cache() {
        assert!(
            Models::new()
                .root()
                .ends_with(".cache/liveinterpreter/models")
        );
    }
}
