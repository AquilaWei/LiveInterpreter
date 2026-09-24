//! Fetching the models the config asks for.
//!
//! Until this existed, [`crate::models::Models::resolve`] failing was the end
//! of the story: it printed where to look and the program stopped. That made
//! `~/.cache/liveinterpreter` undeletable and made a fresh install of the
//! package something that could not actually run. The manifest in
//! `assets/models.toml` has always described what to fetch; this reads it.
//!
//! Three rules shape the code:
//!
//! 1. **A partial file must never look like a model.** Every download lands in
//!    `<name>.part` and is renamed only after its sha256 matches the manifest.
//!    A truncated `model.bin` that got renamed early would fail much later,
//!    inside CTranslate2, as something unrecognisable.
//! 2. **A wrong hash is an error, not a retry.** The manifest's hashes were
//!    computed from the copies this project measured its WER on, so a mismatch
//!    means the file on the other end is not the file the numbers describe.
//!    Quietly accepting it would make every published measurement a claim
//!    about something else.
//! 3. **Resume, because 613 MB.** A `.part` that is already there is continued
//!    with a `Range:` request. Servers that ignore it are handled by starting
//!    over, not by appending to a stale prefix -- which is why the response
//!    status is checked rather than assumed.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use futures_util::StreamExt as _;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::config::EngineConfig;
use crate::models::{Kind, Models};

/// The manifest, compiled in rather than read from disk.
///
/// An installed binary has no `assets/` directory next to it, and a manifest
/// that can go missing is a second way for the program to fail to start --
/// which is the exact failure this module exists to remove.
const MANIFEST: &str = include_str!("../../../assets/models.toml");

#[derive(Debug, Deserialize)]
struct Manifest {
    #[serde(default)]
    model: Vec<Entry>,
}

#[derive(Debug, Deserialize)]
struct Entry {
    id: String,
    kind: String,
    /// Directory under the cache root. `"."` means the root itself, which is
    /// what an archive carrying its own top-level directory wants.
    dest: String,
    /// Prefix for files that do not carry their own `url`.
    #[serde(default)]
    base: Option<String>,
    /// Present only for entries that arrive compressed. `tar.bz2` is the only
    /// value; see the punctuation entry in the manifest for why it is alone.
    #[serde(default)]
    archive: Option<String>,
    /// For an archive: a path under the cache root that exists once it has
    /// been unpacked. An archive's own file is deleted after unpacking, so it
    /// cannot be its own evidence of completeness, and the unpacked names are
    /// not otherwise written down anywhere.
    #[serde(default)]
    provides: Option<String>,
    files: Vec<FileSpec>,
}

#[derive(Debug, Deserialize, Clone)]
struct FileSpec {
    name: String,
    size: u64,
    sha256: String,
    /// Overrides `base`, for a file that does not live with the others.
    #[serde(default)]
    url: Option<String>,
}

/// One file to fetch.
#[derive(Debug, Clone)]
pub struct Item {
    /// The model this belongs to, for a progress line a person can read.
    pub model: String,
    pub name: String,
    pub url: String,
    pub size: u64,
    pub sha256: String,
    /// Where the bytes end up. For an archive this is the tarball itself,
    /// which is unpacked into `unpack_to` and then deleted.
    pub path: PathBuf,
    pub unpack_to: Option<PathBuf>,
}

/// What is missing, and how much it weighs.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub items: Vec<Item>,
    pub total_bytes: u64,
}

impl Plan {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Distinct model ids, in manifest order -- what to show someone before
    /// they agree to a gigabyte.
    pub fn models(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for it in &self.items {
            if !out.contains(&it.model.as_str()) {
                out.push(&it.model);
            }
        }
        out
    }
}

/// Progress, reported per file rather than per byte-batch so a UI can show
/// both "which one" and "how far".
#[derive(Debug, Clone)]
pub enum Progress {
    /// A file is starting. `index` is into [`Plan::items`].
    Started {
        index: usize,
        model: String,
        name: String,
        size: u64,
    },
    /// Bytes for the whole plan, not for this file: a single bar is what a
    /// person actually wants, and per-file progress is recoverable from
    /// `Started` if a caller wants two.
    Bytes {
        done: u64,
        total: u64,
    },
    /// Hashing, which on a 613 MB file is slow enough to look like a hang.
    Verifying {
        name: String,
    },
    Unpacking {
        name: String,
    },
    Finished {
        name: String,
    },
}

/// Which models the config actually needs.
///
/// The degraded accurate-lane model is not in this list unless the config asks
/// for it, so a machine with a GPU never fetches the 57 MB fallback. The
/// punctuation model and the fallback translator are included: both are
/// optional at run time, but someone who is downloading anyway should not have
/// to come back for them.
fn wanted(cfg: &EngineConfig) -> Vec<(Kind, String)> {
    let mut v = Vec::new();
    // Both lanes are optional in the config (`LaneMode::FastOnly` and
    // `AccurateOnly` are real settings), so a lane that is switched off is a
    // lane whose model is not downloaded.
    if let Some(fast) = cfg.asr.fast.as_ref() {
        v.push((Kind::Fast, fast.model.clone()));
        if fast.punctuation {
            v.push((Kind::Punct, fast.punct_model.clone()));
        }
    }
    if let Some(accurate) = cfg.asr.accurate.as_ref() {
        v.push((Kind::Accurate, accurate.model.clone()));
    }
    v.push((Kind::Mt, cfg.mt.model.clone()));
    if !cfg.mt.fallback_model.is_empty() {
        v.push((Kind::Mt, cfg.mt.fallback_model.clone()));
    }
    v
}

fn kind_of(s: &str) -> Option<Kind> {
    match s {
        "fast" => Some(Kind::Fast),
        "accurate" => Some(Kind::Accurate),
        "mt" => Some(Kind::Mt),
        "punct" => Some(Kind::Punct),
        _ => None,
    }
}

/// Work out what is missing without touching the network.
///
/// Completeness is judged **per file, by exact size**, not by asking
/// [`Models::resolve`] whether it can find the model. That distinction is not
/// theoretical: `resolve` answers for a directory-shaped model by checking
/// that the directory exists, and an interrupted download has already created
/// that directory to hold its `.part`. Trusting it meant a killed transfer
/// came back as "All models are already in ...", with a 613 MB `model.bin`
/// that did not exist -- found by doing it (2026-09-12).
///
/// Size rather than sha256 because this runs at every startup: the hash was
/// checked when the bytes arrived, and re-hashing a gigabyte to open a window
/// would be its own bug. Size catches the truncation that actually happens.
pub fn plan(models: &Models, cfg: &EngineConfig) -> Result<Plan> {
    let manifest: Manifest =
        toml::from_str(MANIFEST).context("parsing the built-in assets/models.toml")?;
    let root = models.root();
    let mut plan = Plan::default();

    for (kind, id) in wanted(cfg) {
        // An id that is a path is the user pointing at their own model. They
        // know where it is; we are not going to fetch something else over it.
        if Path::new(&id).is_absolute() || id.contains('/') {
            continue;
        }
        let known = manifest
            .model
            .iter()
            .find(|m| m.id == id && kind_of(&m.kind) == Some(kind));
        // A model that is not in the manifest but is already on disk is
        // someone's own drop-in, and fine. One that is in neither is the error.
        if known.is_none() && models.resolve(kind, &id).is_ok() {
            continue;
        }
        let entry = known.ok_or_else(|| {
            anyhow!(
                "no download for the {} model {id:?}.\n  \
                     assets/models.toml lists: {}\n  \
                     Set `[{}] model` to one of those, or put the files in {} yourself.",
                kind_str(kind),
                manifest
                    .model
                    .iter()
                    .filter(|m| kind_of(&m.kind) == Some(kind))
                    .map(|m| m.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                section(kind),
                root.display()
            )
        })?;

        let dir = if entry.dest == "." {
            root.to_path_buf()
        } else {
            root.join(&entry.dest)
        };
        // An archive is judged by what it leaves behind, since it deletes
        // itself once unpacked.
        if let Some(provides) = &entry.provides
            && root.join(provides).exists()
        {
            continue;
        }
        for f in &entry.files {
            if entry.provides.is_none() && complete(&dir.join(&f.name), f.size) {
                continue;
            }
            let url = match (&f.url, &entry.base) {
                (Some(u), _) => u.clone(),
                (None, Some(b)) => format!("{}/{}", b.trim_end_matches('/'), f.name),
                (None, None) => bail!("{}: {} has no url and the entry has no base", id, f.name),
            };
            plan.total_bytes += f.size;
            plan.items.push(Item {
                model: id.clone(),
                name: f.name.clone(),
                url,
                size: f.size,
                sha256: f.sha256.clone(),
                path: dir.join(&f.name),
                unpack_to: entry.archive.as_ref().map(|_| dir.clone()),
            });
        }
    }
    Ok(plan)
}

fn kind_str(k: Kind) -> &'static str {
    match k {
        Kind::Fast => "fast-lane",
        Kind::Punct => "punctuation",
        Kind::Accurate => "accurate-lane",
        Kind::Mt => "translation",
    }
}

fn section(k: Kind) -> &'static str {
    match k {
        Kind::Fast | Kind::Punct => "asr.fast",
        Kind::Accurate => "asr.accurate",
        Kind::Mt => "mt",
    }
}

/// Fetch everything in `plan`, reporting as it goes.
///
/// Items are fetched in order and one at a time. Parallel downloads would
/// finish sooner on a fast link, but they turn one readable progress bar into
/// four fighting over the same line, and the bottleneck here is usually the
/// other end.
pub async fn fetch(plan: &Plan, mut on: impl FnMut(Progress)) -> Result<()> {
    let client = reqwest::Client::builder()
        .user_agent(concat!("liveinterpreter/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("building the HTTP client")?;

    let mut done: u64 = 0;
    for (index, item) in plan.items.iter().enumerate() {
        on(Progress::Started {
            index,
            model: item.model.clone(),
            name: item.name.clone(),
            size: item.size,
        });
        let before = done;
        get(&client, item, &mut done, plan.total_bytes, &mut on)
            .await
            .with_context(|| format!("fetching {} for {}", item.name, item.model))?;
        // Whatever the transfer did to the running total -- resumed, restarted
        // -- the file is now complete, so the bar reflects that and nothing
        // else.
        done = before + item.size;
        on(Progress::Bytes {
            done,
            total: plan.total_bytes,
        });

        if let Some(dir) = &item.unpack_to {
            on(Progress::Unpacking {
                name: item.name.clone(),
            });
            unpack(&item.path, dir)
                .with_context(|| format!("unpacking {}", item.path.display()))?;
            // The archive is 29 MB of duplicate; keeping it would make the
            // cache look like it holds two copies of the model.
            let _ = fs::remove_file(&item.path);
        }
        on(Progress::Finished {
            name: item.name.clone(),
        });
    }
    Ok(())
}

async fn get(
    client: &reqwest::Client,
    item: &Item,
    done: &mut u64,
    total: u64,
    on: &mut impl FnMut(Progress),
) -> Result<()> {
    if let Some(parent) = item.path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let part = item.path.with_extension(format!(
        "{}part",
        item.path
            .extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default()
    ));

    // A `.part` from an interrupted run is a prefix to continue, not a file to
    // trust: it is still hashed in full at the end.
    let have = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let have = if have > item.size { 0 } else { have };

    let mut req = client.get(&item.url);
    if have > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let resp = req.send().await.context("sending the request")?;
    let status = resp.status();
    if !status.is_success() {
        bail!("{} returned {status}", item.url);
    }
    // 200 to a ranged request means the server ignored the range and is
    // sending the whole file, so the prefix has to go rather than be appended
    // to.
    let resuming = have > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;

    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resuming)
        .truncate(!resuming)
        .open(&part)
        .with_context(|| format!("opening {}", part.display()))?;
    *done += if resuming { have } else { 0 };
    on(Progress::Bytes { done: *done, total });

    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading the response body")?;
        file.write_all(&chunk).context("writing to disk")?;
        *done += chunk.len() as u64;
        on(Progress::Bytes { done: *done, total });
    }
    file.flush().context("flushing to disk")?;
    drop(file);

    on(Progress::Verifying {
        name: item.name.clone(),
    });
    let got = hash(&part)?;
    if !got.eq_ignore_ascii_case(&item.sha256) {
        // Deleted rather than kept: a file that hashes wrong is not a prefix
        // worth resuming, and leaving it would make the next run resume into
        // the same failure for ever.
        let _ = fs::remove_file(&part);
        bail!(
            "{} does not match the manifest.\n  expected sha256 {}\n  got      sha256 {}\n  \
             The file at {} is not the one this project measured on. It has been deleted; \
             running again will fetch it afresh.",
            item.name,
            item.sha256,
            got,
            item.url
        );
    }
    fs::rename(&part, &item.path)
        .with_context(|| format!("renaming into {}", item.path.display()))?;
    Ok(())
}

/// On disk at exactly the size the manifest gives.
///
/// Exactness matters more than it looks: the failure this guards against is a
/// transfer that stopped early, which produces a file that is shorter, and a
/// `>=` would wave it through.
fn complete(path: &Path, size: u64) -> bool {
    fs::metadata(path).is_ok_and(|m| m.len() == size)
}

fn hash(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).with_context(|| format!("hashing {}", path.display()))?;
    Ok(format!("{:x}", h.finalize()))
}

fn unpack(archive: &Path, into: &Path) -> Result<()> {
    let f = fs::File::open(archive).with_context(|| format!("opening {}", archive.display()))?;
    fs::create_dir_all(into).with_context(|| format!("creating {}", into.display()))?;
    let mut tar = tar::Archive::new(bzip2::read::BzDecoder::new(f));
    for entry in tar.entries().context("reading the archive")? {
        let mut entry = entry.context("reading an archive entry")?;
        let path = entry
            .path()
            .context("an archive entry has no path")?
            .into_owned();
        // The archives here are ours, but an archive is still the classic way
        // to be handed `../../etc/something`, and `tar` will happily follow it.
        if path.components().any(|c| {
            matches!(
                c,
                std::path::Component::ParentDir | std::path::Component::RootDir
            )
        }) {
            bail!("refusing an archive entry that escapes the directory: {path:?}");
        }
        entry
            .unpack_in(into)
            .with_context(|| format!("unpacking {path:?}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> EngineConfig {
        EngineConfig::default()
    }

    #[test]
    fn the_builtin_manifest_parses_and_covers_every_default_model() {
        let m: Manifest = toml::from_str(MANIFEST).expect("manifest parses");
        for (kind, id) in wanted(&cfg()) {
            assert!(
                m.model
                    .iter()
                    .any(|e| e.id == id && kind_of(&e.kind) == Some(kind)),
                "no manifest entry for the default {kind:?} model {id:?}"
            );
        }
    }

    #[test]
    fn every_entry_has_a_url_for_every_file() {
        let m: Manifest = toml::from_str(MANIFEST).unwrap();
        for e in &m.model {
            assert!(
                kind_of(&e.kind).is_some(),
                "{}: unknown kind {}",
                e.id,
                e.kind
            );
            assert!(!e.files.is_empty(), "{}: no files", e.id);
            for f in &e.files {
                assert!(
                    f.url.is_some() || e.base.is_some(),
                    "{}: {} has no url and no base",
                    e.id,
                    f.name
                );
                assert_eq!(
                    f.sha256.len(),
                    64,
                    "{}: {} sha256 is not 64 hex",
                    e.id,
                    f.name
                );
                assert!(f.size > 0, "{}: {} has no size", e.id, f.name);
            }
        }
    }

    #[test]
    fn an_empty_cache_needs_everything_and_a_full_one_needs_nothing() {
        let dir = tempdir();
        let models = Models::with_root(&dir);
        let p = plan(&models, &cfg()).unwrap();
        assert_eq!(
            p.models().len(),
            5,
            "fast, accurate, mt, mt fallback, punct"
        );
        assert!(p.total_bytes > 900 * 1024 * 1024, "{}", p.total_bytes);

        // Now satisfy every one of them and ask again.
        for item in &p.items {
            place(item);
        }
        let p = plan(&models, &cfg()).unwrap();
        assert!(p.is_empty(), "still wants {:?}", p.models());
    }

    /// The bug this file's `plan` doc describes, kept honest.
    #[test]
    fn a_half_finished_download_is_not_mistaken_for_a_model() {
        let dir = tempdir();
        let models = Models::with_root(&dir);
        let full = plan(&models, &cfg()).unwrap();
        for item in &full.items {
            place(item);
        }
        assert!(plan(&models, &cfg()).unwrap().is_empty());

        // Interrupt the translation model the way a killed transfer does: the
        // real file gone, a `.part` in its place -- which leaves the directory
        // there, and the directory is all `Models::resolve` looks at.
        let model_bin = dir
            .join("nllb-200-distilled-600m-ct2-int8")
            .join("model.bin");
        fs::remove_file(&model_bin).unwrap();
        fs::write(model_bin.with_extension("bin.part"), b"first 11 MB").unwrap();

        let again = plan(&models, &cfg()).unwrap();
        assert!(
            again.models().contains(&"nllb-200-distilled-600m-ct2-int8"),
            "a .part was accepted as a finished model: {:?}",
            again.models()
        );
        assert!(
            again.items.iter().any(|i| i.name == "model.bin"),
            "the missing file is not the one being fetched: {:?}",
            again.items.iter().map(|i| &i.name).collect::<Vec<_>>()
        );
        // The three files that *are* complete are not fetched again.
        assert_eq!(again.items.len(), 1, "{:?}", again.models());
    }

    #[test]
    fn a_truncated_file_of_the_right_name_is_fetched_again() {
        let dir = tempdir();
        let models = Models::with_root(&dir);
        for item in &plan(&models, &cfg()).unwrap().items {
            place(item);
        }
        let tokens = dir
            .join("sherpa-onnx-streaming-zipformer-en-2023-06-21")
            .join("tokens.txt");
        fs::write(&tokens, b"cut short").unwrap();
        let p = plan(&models, &cfg()).unwrap();
        assert_eq!(p.items.len(), 1);
        assert_eq!(p.items[0].name, "tokens.txt");
    }

    /// Put a file where a finished download would leave it, at the size the
    /// manifest claims. `set_len` makes it sparse, so "613 MB" costs nothing.
    fn place(item: &Item) {
        let path = match &item.unpack_to {
            // The archive deletes itself; what has to exist is what it
            // unpacked, which the manifest names in `provides`.
            Some(dir) => dir.join("sherpa-onnx-online-punct-en-2024-08-06/model.int8.onnx"),
            None => item.path.clone(),
        };
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let f = fs::File::create(&path).unwrap();
        if item.unpack_to.is_none() {
            f.set_len(item.size).unwrap();
        }
    }

    #[test]
    fn a_model_id_that_is_a_path_is_the_users_own_and_is_not_fetched() {
        let dir = tempdir();
        let mut c = cfg();
        c.mt.model = "/opt/models/my-own-nllb".into();
        let p = plan(&Models::with_root(&dir), &c).unwrap();
        assert!(
            !p.models().contains(&"/opt/models/my-own-nllb"),
            "{:?}",
            p.models()
        );
    }

    #[test]
    fn an_unknown_model_id_says_what_is_on_offer() {
        let dir = tempdir();
        let mut c = cfg();
        c.asr.accurate.as_mut().unwrap().model = "enormous.en-q9_9".into();
        let err = plan(&Models::with_root(&dir), &c).unwrap_err().to_string();
        assert!(err.contains("enormous.en-q9_9"), "{err}");
        assert!(err.contains("small.en-q5_1"), "{err}");
        assert!(err.contains("asr.accurate"), "{err}");
    }

    #[test]
    fn the_degraded_model_is_only_fetched_when_it_is_asked_for() {
        let dir = tempdir();
        let plain = plan(&Models::with_root(&dir), &cfg()).unwrap();
        assert!(!plain.models().contains(&"base.en-q5_1"));

        let mut c = cfg();
        c.asr.accurate.as_mut().unwrap().model = "base.en-q5_1".into();
        let degraded = plan(&Models::with_root(&dir), &c).unwrap();
        assert!(degraded.models().contains(&"base.en-q5_1"));
        assert!(!degraded.models().contains(&"small.en-q5_1"));
    }

    #[test]
    fn the_fallback_translator_is_fetched_with_the_rest() {
        let dir = tempdir();
        let p = plan(&Models::with_root(&dir), &cfg()).unwrap();
        assert!(p.models().contains(&"opus-mt-en-zh-ct2-int8"));
    }

    #[test]
    fn a_fallback_translator_turned_off_is_not_fetched() {
        let dir = tempdir();
        let mut c = cfg();
        c.mt.fallback_model = String::new();
        let p = plan(&Models::with_root(&dir), &c).unwrap();
        assert!(!p.models().contains(&"opus-mt-en-zh-ct2-int8"));
    }

    #[test]
    fn an_archive_entry_lands_in_the_root_and_knows_to_unpack() {
        let dir = tempdir();
        let p = plan(&Models::with_root(&dir), &cfg()).unwrap();
        let punct = p
            .items
            .iter()
            .find(|i| i.model == "online-punct-en-2024-08-06")
            .expect("punct is in the plan");
        assert_eq!(punct.unpack_to.as_deref(), Some(dir.as_path()));
        assert_eq!(punct.path.parent(), Some(dir.as_path()));
    }

    #[test]
    fn an_archive_that_climbs_out_of_the_directory_is_refused() {
        let dir = tempdir();
        let archive = dir.join("evil.tar.bz2");
        {
            let f = fs::File::create(&archive).unwrap();
            let enc = bzip2::write::BzEncoder::new(f, bzip2::Compression::fast());
            let mut b = tar::Builder::new(enc);
            let mut header = tar::Header::new_gnu();
            header.set_size(3);
            header.set_mode(0o644);
            // The name goes straight into the raw header: `set_path` and
            // `append_data` both refuse `..`, so building the archive any
            // other way would test the tar crate's writer rather than this
            // module's reader. A real hostile archive is written by something
            // that does not ask permission.
            {
                let name = b"../escaped.txt";
                let gnu = header.as_gnu_mut().expect("a gnu header");
                gnu.name[..name.len()].copy_from_slice(name);
            }
            header.set_cksum();
            b.append(&header, &b"bad"[..]).unwrap();
            b.into_inner().unwrap().finish().unwrap();
        }
        let err = unpack(&archive, &dir.join("into")).unwrap_err().to_string();
        assert!(err.contains("escapes"), "{err}");
        assert!(!dir.join("escaped.txt").exists());
    }

    #[test]
    fn hashing_a_known_file_matches_sha256sum() {
        let dir = tempdir();
        let f = dir.join("x");
        fs::write(&f, b"abc").unwrap();
        assert_eq!(
            hash(&f).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// A scratch directory that cleans itself up is not worth a dependency for
    /// the handful of tests here; the pid keeps concurrent runs apart.
    fn tempdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "li-download-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }
}
