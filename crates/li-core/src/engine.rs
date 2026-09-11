//! Pipeline assembly (PLAN §10.3, §11).
//!
//! Every other crate does one job and has no opinion about the others. This is
//! where they become a program.
//!
//! ## Why this is not `xtask eval` with a microphone
//!
//! The measurement harness runs the whole pipeline in one synchronous loop:
//! read a frame, feed the fast lane, run whisper, repeat. Over a file that is
//! fine and it makes the numbers easy to reason about. Live it would be wrong,
//! because a whisper pass takes 200-500 ms and nothing would be reading the
//! capture stream while it ran -- the audio would pile up and then be dropped,
//! which is the one failure the two-lane design exists to avoid.
//!
//! So (PLAN §11):
//!
//! ```text
//!  [capture thread] --frames--> [pipeline] --segments--> [accurate worker]
//!    (li-audio)                    │  ▲                   (whisper, blocking)
//!                                  │  └───hypotheses──────────┘
//!                                  ├──lines──> [MT worker] ──translations──┐
//!                                  │            (NLLB, blocking)           │
//!                                  └──events──> [sink] <────────────────────┘
//!                                                 │  (transcript files)
//!                                                 └──> broadcast --> UI
//! ```
//!
//! The fast lane stays in the pipeline: it is chunk-based and cheap, and
//! keeping it next to the frames is what makes its endpoints -- which is to say
//! the sentence boundaries everything downstream uses -- arrive on time.
//!
//! **Backpressure.** The segment channel is short. If the accurate lane falls
//! behind far enough to fill it, the segment is dropped and the line it belongs
//! to is promoted from the fast lane by the ordinary 8 s rule (PLAN §12.2) --
//! degraded, but never a hole and never a growing queue. Finalised *text* is
//! never dropped: the sink channel blocks instead.
//!
//! **Shutdown** works by closing channels rather than by a flag anybody has to
//! remember to check. Dropping the audio source ends the capture thread, which
//! ends the frame stream, which ends the pipeline, which drops the two senders
//! it holds, which ends the workers, which ends the sink -- and the sink is
//! what calls `Writer::finish`, so the transcript is closed properly exactly
//! once, at the end of the chain.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use li_asr::{AsrEngine, punct::OnlinePunct, window::Ring};
use li_audio::{AudioSource, DesktopSource};
use li_stream::{LaneMode, Stream, StreamConfig};
use li_transcript::{Langs, TranscriptSink, Writer};
use li_types::{
    AsrEvent, AudioFrame, EngineEvent, EngineStatus, Lane, LatencySample, Stage, TranscriptLine,
    Word,
};
use li_vad::{GateConfig, SileroVad, Vad, VadEvent};
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;

use crate::config::{EngineConfig, LaneCfg};
use crate::models::{Kind, Models};

const SAMPLE_RATE: f64 = 16_000.0;

/// How many utterances may be waiting for the accurate lane.
///
/// Two, not twenty: a deep queue would let the accurate lane fall further and
/// further behind while still answering, and its answers would arrive after the
/// lines they belong to had already been promoted and written. Short queue,
/// visible degradation.
const SEGMENT_QUEUE: usize = 2;

/// Lines waiting to be translated. Deep enough that a burst of short sentences
/// does not lose one, shallow enough that a machine which cannot keep up drops
/// the oldest instead of falling further behind. Drafting from the fast lane
/// puts up to two entries in here per line, which is what the 8 already allowed
/// for.
///
/// Depth alone is not the whole policy. NLLB is 0.36-0.9 s a line on this
/// machine (task 1.19) and runs on one thread, so a queue that is merely bounded
/// still translates lines in the order they were spoken while the bar shows a
/// sentence from ten seconds ago. [`drop_superseded_drafts`] is the other half:
/// what a live subtitle owes the reader is the newest line, not every line.
const MT_QUEUE: usize = 8;

/// One utterance on its way to the accurate lane.
struct Segment {
    pcm: Vec<f32>,
    t_origin: Duration,
    prompt: String,
}

/// Anything the sink has to act on. One channel rather than two, so the sink
/// exits exactly when every producer has.
enum SinkMsg {
    Event(EngineEvent),
    Translation {
        line_id: u64,
        text: String,
        /// False for the draft made from fast-lane text. It reaches the screen
        /// and stops there; the transcript takes the settled one.
        settled: bool,
        audio_end: Instant,
        lane: Lane,
    },
}

/// One line handed to the translator.
///
/// `settled` is what the text is, not how urgent it is: the draft is the fast
/// lane's own sentence, the settled one is whatever `li-stream` finalised. Both
/// carry the same `line_id`, and they are translated in the order they were
/// queued, so the draft can never overwrite the answer that replaced it.
struct MtJob {
    line_id: u64,
    text: String,
    settled: bool,
    /// A guess made while the line was still open, translated during the
    /// silence that ends it. It never reaches the screen on its own: it is put
    /// in [`Drafts`] and used only if the line then closes saying exactly what
    /// was translated. See [`Speculation`].
    speculative: bool,
    /// Wall clock for the end of this line's audio, so the sink can report how
    /// long the reader actually waited. PLAN §7 measures latency from there,
    /// not from when the line was queued.
    audio_end: Instant,
    lane: Lane,
}

/// How long the fast lane's words must stop moving before they are translated
/// on spec.
///
/// 350 ms, from the measurement in task 1.20. On `ami_meeting.wav` the fast
/// lane's last change to a line lands 631-647 ms before that line closes -- a
/// very tight band, because what fills it is `rule2_min_trailing_silence`
/// counting out its 0.6 s. Mid-sentence pauses long enough to be mistaken for
/// that are rarer than they feel: 18 per 100 s against 32 real sentences, and
/// none of them fell between 350 and 450 ms, so the threshold is not balanced
/// on an edge.
const SPECULATE_AFTER: Duration = Duration::from_millis(350);

/// Translate the fast lane's sentence during the silence that ends it.
///
/// NLLB is the largest single term in how long Chinese takes to appear (task
/// 1.19: 0.3-0.9 s a line, one thread, and it scales with the sentence), and
/// for the 0.6 s before a line closes it has nothing to do while the words
/// themselves have already stopped changing. Handing it the text then puts the
/// answer in [`Drafts`] before the line is even closed, so the draft costs
/// whatever is left of the pass instead of the whole of it.
///
/// A guess is never shown. It reaches the bar only through `Drafts`, when the
/// line closes saying character-for-character what was translated -- so a guess
/// made at a mid-sentence pause costs CPU and nothing else. That is what makes
/// this safe to do on a hypothesis: being wrong is invisible.
#[derive(Default)]
struct Speculation(Option<Guess>);

struct Guess {
    line_id: u64,
    text: String,
    /// When the words last moved. Wall clock, because what is being waited on
    /// is the model being idle, not a position in the audio.
    since: Instant,
    sent: bool,
}

impl Speculation {
    /// The fast lane revised the line, or started a new one. The clock restarts:
    /// text that is still moving is not worth a pass.
    ///
    /// Raw string equality, still, after task 1.25 put punctuation on this
    /// text. The plan for 1.25 expected to have to weaken this to content
    /// words, on the grounds that a full stop appearing and disappearing would
    /// restart the clock forever and quietly delete task 1.20's whole benefit.
    /// It does not, because of *where* the marks are added:
    /// [`crate::engine::punctuate`] is a deterministic function of the raw
    /// hypothesis, and the raw hypothesis has no marks and no casing at all. So
    /// punctuated partials differ exactly when raw ones did -- the comparison
    /// sees the same sequence of changes it saw before. Weakening it would
    /// instead introduce a bug: two hypotheses that agree on content words but
    /// not on marks would keep the older text, and `Drafts::reuse` would then
    /// miss the line it was translated for. If stage 2 ever punctuates from
    /// somewhere other than the raw text, this reasoning expires with it.
    fn observe(&mut self, line_id: u64, text: &str) {
        if self
            .0
            .as_ref()
            .is_some_and(|g| g.line_id == line_id && g.text == text)
        {
            return;
        }
        self.0 = Some(Guess {
            line_id,
            text: text.to_owned(),
            since: Instant::now(),
            sent: false,
        });
    }

    /// The line closed, or was finalised. There is nothing left to guess at,
    /// and the real draft is already on its way.
    fn settle(&mut self) {
        self.0 = None;
    }

    /// Queue the guess once the words have been still for [`SPECULATE_AFTER`].
    /// Called once a frame, which is every 32 ms -- fine, because the work is
    /// one comparison until the moment it is not.
    fn tick(&mut self, out: &Out) {
        let Some(g) = self.0.as_mut() else { return };
        if g.sent || g.since.elapsed() < SPECULATE_AFTER {
            return;
        }
        // Marked either way: a line of backchannel does not become worth
        // translating by being looked at again 32 ms later.
        g.sent = true;
        if !worth_translating(&g.text) {
            return;
        }
        let _ = out.mt.try_send(MtJob {
            line_id: g.line_id,
            text: g.text.clone(),
            settled: false,
            speculative: true,
            // Never used: a guess is not sent to the sink, so nothing measures
            // how long a reader waited for it.
            audio_end: Instant::now(),
            lane: Lane::Fast,
        });
    }
}

/// Where finalised events and translatable lines go.
///
/// The two always travel together -- every finalised line is both -- and
/// bundling them keeps the flag that decides whether drafts are sent next to
/// the channel it is a decision about.
struct Out {
    sink: mpsc::Sender<SinkMsg>,
    mt: mpsc::Sender<MtJob>,
    drafts: bool,
}

/// Converts a position on the audio timeline into wall time.
///
/// Latency is "the audio for this sentence ends" -> "the text is on screen"
/// (PLAN §7), and the two ends of that interval are kept by different clocks:
/// the engine counts samples, the reader waits in seconds. The offset between
/// them is not a constant -- `li-audio` drops frames when the pipeline stalls,
/// and every dropped frame moves the audio timeline earlier against the wall
/// -- so it is re-derived from each frame's own capture instant rather than
/// anchored once at the start.
///
/// Lives on the pipeline thread, which is the only thread that sees frames.
#[derive(Debug, Clone, Copy, Default)]
struct Clock(Option<(Instant, Duration)>);

impl Clock {
    /// `t_capture` is when this frame was handed over; `audio_end` is where it
    /// ends on the audio timeline.
    fn observe(&mut self, t_capture: Instant, audio_end: Duration) {
        self.0 = Some((t_capture, audio_end));
    }

    /// Wall time for a position on the audio timeline.
    fn wall(&self, at: Duration) -> Instant {
        match self.0 {
            // `t_capture` is the wall time this frame's audio *ended*, near
            // enough: `li-audio` stamps a frame as it hands it over.
            Some((t_capture, audio_end)) if at >= audio_end => t_capture + (at - audio_end),
            Some((t_capture, audio_end)) => t_capture - (audio_end - at),
            // No frame yet, so there is nothing to measure against and the only
            // honest answer is "now" -- a latency of zero rather than a made-up
            // one.
            None => Instant::now(),
        }
    }

    /// How far behind the live audio the newest words are, right now.
    ///
    /// Everything the reader feels is in here: the capture device's own
    /// buffering, the resampler, the VAD, and the fast lane's decode. It is
    /// the number to look at first when the bar feels slow, because a lag that
    /// is already a second before any text is produced cannot be fixed
    /// anywhere downstream.
    fn lag(&self) -> Duration {
        match self.0 {
            Some((t_capture, _)) => t_capture.elapsed(),
            None => Duration::ZERO,
        }
    }
}

pub struct Engine {
    cfg: EngineConfig,
    events: broadcast::Sender<EngineEvent>,
    session: Option<Session>,
}

/// A running session. Dropping `capture` is what starts the shutdown chain.
struct Session {
    /// The capture backend, held only so that dropping it stops the device.
    /// Boxed because a file source has nothing to stop and no common trait
    /// says so -- `AudioSource` describes opening, not closing.
    capture: Box<dyn Send>,
    paused: Arc<AtomicBool>,
    tasks: Vec<JoinHandle<()>>,
    transcripts: Vec<std::path::PathBuf>,
}

/// The models and files a session needs, loaded but not yet running.
struct Loaded {
    fast: Option<Box<dyn AsrEngine>>,
    /// The fast lane's punctuation and casing, if this build has the model.
    /// `None` is a supported state, not a failure -- see [`load_punct`].
    punct: Option<OnlinePunct>,
    accurate: Option<Box<dyn AsrEngine>>,
    mt: Option<li_mt::LocalNllb>,
    writer: Writer,
    mode: LaneMode,
}

impl Engine {
    pub fn new(cfg: EngineConfig) -> Result<Self> {
        let (events, _) = broadcast::channel(256);
        Ok(Self {
            cfg,
            events,
            session: None,
        })
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EngineEvent> {
        self.events.subscribe()
    }

    pub fn is_running(&self) -> bool {
        self.session.is_some()
    }

    /// The transcript files this session is writing (PLAN §13.1).
    pub fn transcripts(&self) -> &[std::path::PathBuf] {
        self.session.as_ref().map_or(&[], |s| &s.transcripts)
    }

    /// Load the models, open the capture device, and start the pipeline.
    ///
    /// Model loading is reported as it happens: whisper on a cold GPU takes
    /// seconds, and a UI with no status line looks broken for all of them. It
    /// also happens *before* the device is opened, so the first thing anyone
    /// says is not thrown away while the GPU warms up.
    pub async fn start(&mut self) -> Result<()> {
        let loaded = self.load().await?;
        let mut source = DesktopSource::default();
        let frames = source
            .open(self.cfg.audio.source.clone())
            .await
            .context("opening the capture device")?;
        self.launch(loaded, frames, Box::new(source))
    }

    /// The same pipeline, driven by a wav file at playback speed.
    ///
    /// Not a test convenience: it is the only way to exercise the assembled
    /// engine repeatably, on audio whose right answer is known. The file source
    /// paces itself, so a lane that cannot keep up loses audio exactly as it
    /// would in a meeting.
    pub async fn start_from_wav(&mut self, wav: &std::path::Path) -> Result<()> {
        let loaded = self.load().await?;
        let frames = li_audio::file::FileSource::new(wav)
            .with_agc(true)
            .open()
            .with_context(|| format!("opening {}", wav.display()))?;
        self.launch(loaded, frames, Box::new(()))
    }

    async fn load(&mut self) -> Result<Loaded> {
        if self.session.is_some() {
            bail!("the engine is already running");
        }
        let cfg = self.cfg.clone();
        let models = Models::new();
        let status = |s: EngineStatus| {
            let _ = self.events.send(EngineEvent::Status(s));
        };

        // --- models, off the runtime: each of these blocks for seconds ---
        let mode = lane_mode(&cfg)?;
        let fast = match (&cfg.asr.fast, mode) {
            (Some(lane), LaneMode::Dual | LaneMode::FastOnly) => {
                status(EngineStatus::ModelLoading {
                    what: format!("fast lane ({})", lane.model),
                });
                let spec = lane.to_spec(&models, Kind::Fast, &cfg.asr.language)?;
                Some(tokio::task::spawn_blocking(move || li_asr::build(&spec)).await??)
            }
            _ => None,
        };
        let punct = match (&cfg.asr.fast, fast.is_some()) {
            (Some(lane), true) => load_punct(lane, &models, &cfg.asr.language),
            // No fast lane, nothing to punctuate: `AccurateOnly` gets its
            // marks from whisper, which writes its own.
            _ => None,
        };
        let accurate = match (&cfg.asr.accurate, mode) {
            (Some(lane), LaneMode::Dual | LaneMode::AccurateOnly) => {
                status(EngineStatus::ModelLoading {
                    what: format!("accurate lane ({})", lane.model),
                });
                let spec = lane.to_spec(&models, Kind::Accurate, &cfg.asr.language)?;
                Some(tokio::task::spawn_blocking(move || li_asr::build(&spec)).await??)
            }
            _ => None,
        };
        let mt = match cfg.mt.backend.as_str() {
            "off" => None,
            "local" => {
                status(EngineStatus::ModelLoading {
                    what: format!("translation ({})", cfg.mt.model),
                });
                let nllb = cfg.mt.to_nllb(&models)?;
                Some(tokio::task::spawn_blocking(move || li_mt::LocalNllb::open(&nllb)).await??)
            }
            other => bail!("mt backend {other:?} is not in this build (want `local` or `off`)"),
        };

        for e in [fast.as_ref(), accurate.as_ref()].into_iter().flatten() {
            tracing::info!("{}", e.backend());
        }

        let writer = Writer::open(&cfg.transcript, &Langs::default())
            .context("opening the transcript files")?;
        for p in writer.paths() {
            tracing::info!("transcript: {}", p.display());
        }

        Ok(Loaded {
            fast,
            punct,
            accurate,
            mt,
            writer,
            mode,
        })
    }

    fn launch(
        &mut self,
        loaded: Loaded,
        frames: mpsc::Receiver<AudioFrame>,
        capture: Box<dyn Send>,
    ) -> Result<()> {
        let Loaded {
            fast,
            punct,
            accurate,
            mt,
            writer,
            mode,
        } = loaded;
        let transcripts = writer.paths().to_vec();

        // --- wiring ---
        let (seg_tx, seg_rx) = mpsc::channel::<Segment>(SEGMENT_QUEUE);
        let (hyp_tx, hyp_rx) = mpsc::channel::<Vec<Word>>(SEGMENT_QUEUE + 2);
        let (sink_tx, sink_rx) = mpsc::channel::<SinkMsg>(64);
        let (mt_tx, mt_rx) = mpsc::channel::<MtJob>(MT_QUEUE);
        let paused = Arc::new(AtomicBool::new(false));

        let mut tasks = Vec::new();
        if let Some(acc) = accurate {
            tasks.push(spawn_accurate(acc, seg_rx, hyp_tx));
        }
        if let Some(mt) = mt {
            tasks.push(spawn_mt(mt, mt_rx, sink_tx.clone()));
        }
        tasks.push(spawn_sink(writer, sink_rx, self.events.clone()));
        tasks.push(spawn_pipeline(Pipeline {
            cfg: self.cfg.stream(),
            mode,
            fast,
            punct,
            frames,
            seg_tx,
            hyp_rx,
            out: Out {
                sink: sink_tx,
                mt: mt_tx,
                // Only dual-lane mode can produce a draft at all: with the
                // accurate lane off, `li-stream` finalises the fast lane's
                // sentence outright and there is nothing to be early about.
                drafts: self.cfg.mt.draft_from_fast_lane && mode == LaneMode::Dual,
            },
            paused: paused.clone(),
        })?);

        let _ = self.events.send(EngineEvent::Status(EngineStatus::Running));
        self.session = Some(Session {
            capture,
            paused,
            tasks,
            transcripts,
        });
        Ok(())
    }

    /// Stop reading audio. The models stay loaded, so resuming is instant.
    pub fn pause(&self, on: bool) {
        if let Some(s) = &self.session {
            s.paused.store(on, Ordering::Relaxed);
            let _ = self.events.send(EngineEvent::Status(if on {
                EngineStatus::Paused
            } else {
                EngineStatus::Running
            }));
        }
    }

    /// End the session and wait for the transcript to be closed.
    ///
    /// Dropping the source stops the capture thread; everything downstream then
    /// finishes in order because its input channel closed. Waiting for the
    /// tasks is what makes "stopped" mean the files are complete.
    pub async fn stop(&mut self) -> Result<()> {
        let Some(session) = self.session.take() else {
            return Ok(());
        };
        drop(session.capture);
        for t in session.tasks {
            if let Err(e) = t.await {
                tracing::warn!("a pipeline task ended badly: {e}");
            }
        }
        Ok(())
    }
}

/// Which lanes the config actually asks for.
fn lane_mode(cfg: &EngineConfig) -> Result<LaneMode> {
    let on = |l: &Option<crate::config::LaneCfg>| l.as_ref().is_some_and(|l| l.backend != "off");
    Ok(match (on(&cfg.asr.fast), on(&cfg.asr.accurate)) {
        (true, true) => LaneMode::Dual,
        (true, false) => LaneMode::FastOnly,
        (false, true) => LaneMode::AccurateOnly,
        (false, false) => bail!(
            "both ASR lanes are off -- set `[asr.fast]` or `[asr.accurate]` to something \
             other than `off` (PLAN §8.2)"
        ),
    })
}

/// The accurate lane, on a thread of its own.
///
/// `spawn_blocking` rather than a plain task: whisper holds the CPU (or blocks
/// on the GPU) for the whole pass, and doing that on a runtime worker would
/// stall every other task on it.
fn spawn_accurate(
    mut acc: Box<dyn AsrEngine>,
    mut segments: mpsc::Receiver<Segment>,
    hypotheses: mpsc::Sender<Vec<Word>>,
) -> JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        while let Some(seg) = segments.blocking_recv() {
            acc.set_prompt(&seg.prompt);
            let words = match rt.block_on(acc.feed(&seg.pcm, seg.t_origin)) {
                Ok(evs) => evs
                    .into_iter()
                    .find_map(|e| match e {
                        AsrEvent::Partial { words, .. } => Some(words),
                        _ => None,
                    })
                    .unwrap_or_default(),
                Err(e) => {
                    tracing::warn!("accurate lane: {e:#}");
                    continue;
                }
            };
            if hypotheses.blocking_send(words).is_err() {
                break;
            }
        }
    })
}

fn spawn_mt(
    mt: li_mt::LocalNllb,
    mut lines: mpsc::Receiver<MtJob>,
    sink: mpsc::Sender<SinkMsg>,
) -> JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        let mut drafted = Drafts::default();
        // Everything that arrived while the last line was in the model. The
        // queue cannot be read one job at a time: deciding whether a draft is
        // still worth 0.7 s means knowing what is behind it.
        let mut pending: VecDeque<MtJob> = VecDeque::new();
        loop {
            if pending.is_empty() {
                match lines.blocking_recv() {
                    Some(job) => pending.push_back(job),
                    None => break,
                }
            }
            while let Ok(job) = lines.try_recv() {
                pending.push_back(job);
            }
            drop_superseded_drafts(&mut pending);
            let MtJob {
                line_id,
                text: src,
                settled,
                speculative,
                audio_end,
                lane,
            } = pending.pop_front().expect("not empty");
            let began = Instant::now();
            let out = match drafted.reuse(line_id, &src, settled) {
                Some(text) => Ok(text),
                // Which lane wrote the marks decides whether they may be the
                // only place the line is cut. Since task 1.25 both lanes
                // punctuate, but only whisper *heard* what it wrote: the fast
                // lane's marks are restored from the words, and trusting them
                // alone took the fast lane's translation from 1.48 Chinese
                // characters per English word to 1.08 (`li_mt::chunk::Marks`).
                None => mt.translate_blocking(
                    &src,
                    match lane {
                        Lane::Accurate => li_mt::chunk::Marks::Heard,
                        Lane::Fast => li_mt::chunk::Marks::Restored,
                    },
                ),
            };
            // The term that dominates the translation row, written down where
            // the source row's is. Measured on this machine: 357 ms for a
            // six-word line, 788 ms for eighteen -- it scales with the
            // sentence, so it is decode time, not a fixed cost that could be
            // tuned away (task 1.19).
            tracing::info!(
                line_id,
                settled,
                speculative,
                words = src.split_whitespace().count(),
                ms = began.elapsed().as_millis() as u64,
                "translated"
            );
            match out {
                Ok(text) => {
                    if !settled {
                        drafted.remember(line_id, src, &text);
                    }
                    // A guess stops here. It is worth having only if the line
                    // closes saying exactly this, and then the draft that
                    // follows finds it in `Drafts` and costs nothing.
                    if speculative {
                        continue;
                    }
                    if sink
                        .blocking_send(SinkMsg::Translation {
                            line_id,
                            text,
                            settled,
                            audio_end,
                            lane,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
                // A line that will not translate is a line the subtitle shows in
                // English. It is not a reason to stop interpreting.
                Err(e) => tracing::warn!("translating line {line_id}: {e:#}"),
            }
        }
    })
}

/// Throw away the drafts that something newer has already overtaken.
///
/// A draft exists only to be early. Once a newer line is waiting -- or this
/// line's own settled text is -- it cannot be early any more, and translating
/// it would spend the one MT thread putting Chinese on the bar that the very
/// next job overwrites. NLLB is 0.36-0.9 s a line and runs sequentially, so
/// under continuous speech that wasted pass is the difference between the
/// translation row keeping up and falling further behind for as long as
/// somebody talks. This is where drafting is made free: it can delay a settled
/// translation by at most the pass already in the model.
///
/// Settled lines are never dropped here. They are what `on_translation_final`
/// writes, and PLAN §11 does not let finalised text disappear.
fn drop_superseded_drafts(pending: &mut VecDeque<MtJob>) {
    // How much a job is worth keeping, in the order a queue should give it up:
    // a guess is the first thing to go, a settled line never goes at all.
    let rank = |j: &MtJob| (j.line_id, if j.settled { 2 } else { !j.speculative as u8 });
    let Some(best) = pending.iter().map(rank).max() else {
        return;
    };
    let before = pending.len();
    pending.retain(|j| j.settled || rank(j) == best);
    if pending.len() < before {
        tracing::debug!(
            dropped = before - pending.len(),
            queued = pending.len(),
            "drafts overtaken"
        );
    }
}

/// What each draft was translated from, and what it came back as.
///
/// A line the accurate lane never answered for is promoted with the fast lane's
/// own words, so its settled text is character-for-character the draft's --
/// and running NLLB over it again would spend 175 ms to produce what is already
/// on screen. The match is on the text, not on the id alone: when the accurate
/// lane *did* answer, the words differ and the line is translated properly.
///
/// Bounded by the queue it shadows, so a draft whose settled line never arrives
/// falls off the front rather than accumulating for the length of a meeting.
#[derive(Default)]
struct Drafts(VecDeque<(u64, String, String)>);

impl Drafts {
    fn remember(&mut self, line_id: u64, src: String, text: &str) {
        self.0.push_back((line_id, src, text.to_owned()));
        while self.0.len() > MT_QUEUE {
            self.0.pop_front();
        }
    }

    /// An answer already worked out for exactly these words on exactly this
    /// line.
    ///
    /// Two things arrive here. A settled line matches when the accurate lane
    /// never answered and the line was promoted with the fast lane's own words
    /// -- then re-running NLLB would spend a pass reproducing what is already
    /// on screen. A draft matches when [`Speculation`] guessed this sentence
    /// during the silence that closed it, which is the common case and the
    /// point of guessing.
    ///
    /// A settled line clears whatever else was remembered for it either way:
    /// there is one settled line per `line_id`, and a guess it did not match is
    /// a guess that was wrong.
    fn reuse(&mut self, line_id: u64, src: &str, settled: bool) -> Option<String> {
        let hit = self
            .0
            .iter()
            .position(|(id, draft, _)| *id == line_id && draft == src)
            .map(|i| self.0.remove(i).expect("just found").2);
        if settled {
            self.0.retain(|(id, _, _)| *id != line_id);
        }
        hit
    }
}

/// The one place a latency reading is written down.
///
/// At `info`, because it is the answer to "why does this feel slow" and that
/// question is always asked after the fact, from a log somebody already has --
/// `journalctl --user` for the packaged app. One line per finalised line and
/// per translation is a few a minute, not a stream.
fn log_latency(m: &LatencySample) {
    tracing::info!(
        line_id = m.line_id,
        stage = ?m.stage,
        lane = ?m.lane,
        settled = m.settled,
        ms = m.latency.as_millis() as u64,
        "latency"
    );
}

/// Owns the transcript files and the broadcast to the UI.
///
/// One place writes the files, so `Writer` never has to be shared, and
/// `finish()` happens exactly once -- when every producer has gone.
fn spawn_sink(
    mut writer: Writer,
    mut rx: mpsc::Receiver<SinkMsg>,
    events: broadcast::Sender<EngineEvent>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match msg {
                SinkMsg::Event(ev) => {
                    if let EngineEvent::SourceFinal {
                        line_id,
                        text,
                        t_start,
                        t_end,
                        lane,
                        reason,
                    } = &ev
                        && let Err(e) = writer.on_source_final(&TranscriptLine {
                            line_id: *line_id,
                            start_s: t_start.as_secs_f64(),
                            end_s: t_end.as_secs_f64(),
                            source: text.clone(),
                            translation: None,
                            lane: *lane,
                            reason: *reason,
                        })
                    {
                        tracing::error!("writing the transcript: {e:#}");
                    }
                    if let EngineEvent::Metrics(m) = &ev {
                        log_latency(m);
                    }
                    let _ = events.send(ev);
                }
                SinkMsg::Translation {
                    line_id,
                    text,
                    settled,
                    audio_end,
                    lane,
                } => {
                    // A draft goes to the screen only. `on_translation_final`
                    // is a fill-in for a line already written and the bilingual
                    // file drains past it once called, so calling it twice for
                    // one line would leave the earlier, worse text in the file.
                    if settled && let Err(e) = writer.on_translation_final(line_id, &text) {
                        tracing::error!("writing the transcript: {e:#}");
                    }
                    let _ = events.send(EngineEvent::Translation {
                        line_id,
                        text,
                        settled,
                    });
                    let sample = LatencySample {
                        line_id,
                        lane,
                        stage: Stage::Translation,
                        settled,
                        latency: audio_end.elapsed(),
                    };
                    log_latency(&sample);
                    let _ = events.send(EngineEvent::Metrics(sample));
                }
            }
        }
        if let Err(e) = writer.finish() {
            tracing::error!("closing the transcript: {e:#}");
        }
        // The sink is the last link in the chain, so this is the one place that
        // can honestly say the session is over and the files are complete --
        // including when nobody asked for it, because the capture device went
        // away or the wav ran out.
        let _ = events.send(EngineEvent::Status(EngineStatus::Stopped));
    })
}

/// Open the fast lane's punctuation model, or explain once why there is none.
///
/// Never an error. Both ASR lanes go through `Models::resolve` and fail
/// startup when their model is missing, because a build with no recogniser is
/// not this program; this one is an improvement to text that is already
/// correct, so a machine without the 7.5 MB download runs exactly as it did
/// before task 1.25 and says so at `info`. PLAN §11 lists it as optional and
/// PLAN §15 has the URL, which is what the message points at.
fn load_punct(lane: &LaneCfg, models: &Models, language: &str) -> Option<OnlinePunct> {
    if !lane.punctuation {
        return None;
    }
    // The model is English-only, and so is the fast lane's
    // (`li_asr::LaneSpec::language`). Silence rather than an error: someone who
    // switches language has not misconfigured anything.
    if !language.starts_with("en") {
        tracing::info!("punctuation: off, the model is English only (language = {language:?})");
        return None;
    }
    let dir = match models.resolve(Kind::Punct, &lane.punct_model) {
        Ok(dir) => dir,
        Err(e) => {
            tracing::info!("punctuation: off ({e:#})");
            return None;
        }
    };
    // Two threads, like the fast lane it shares a thread with: this is a
    // 7.5 MB int8 CNN answering in about 5 ms, and the cores are wanted by
    // whisper.
    match OnlinePunct::open(&dir, 2) {
        Ok(p) => Some(p),
        Err(e) => {
            tracing::warn!("punctuation: off ({e:#})");
            None
        }
    }
}

struct Pipeline {
    cfg: StreamConfig,
    mode: LaneMode,
    fast: Option<Box<dyn AsrEngine>>,
    /// Lives on the pipeline thread with the fast lane, reached only through
    /// `&mut`, never shared -- the same rules as `SherpaFast`, for the same
    /// reason (`sherpa.rs:79-82`).
    punct: Option<OnlinePunct>,
    frames: mpsc::Receiver<AudioFrame>,
    seg_tx: mpsc::Sender<Segment>,
    hyp_rx: mpsc::Receiver<Vec<Word>>,
    out: Out,
    paused: Arc<AtomicBool>,
}

fn spawn_pipeline(mut p: Pipeline) -> Result<JoinHandle<()>> {
    let mut vad = SileroVad::new(GateConfig::default())?;
    Ok(tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let mut stream = Stream::new(p.cfg, p.mode);
        let mut ring = Ring::default();
        let pause_flush = Duration::from_secs_f64(p.cfg.pause_flush_s);
        let max_segment = Duration::from_secs_f64(p.cfg.max_segment_s);
        let mut samples: u64 = 0;
        let mut speech_start: Option<Duration> = None;
        let mut clock = Clock::default();
        let mut spec = Speculation::default();
        let mut lag_reported = Instant::now();

        while let Some(frame) = p.frames.blocking_recv() {
            if p.paused.load(Ordering::Relaxed) {
                continue;
            }
            let t_origin = Duration::from_secs_f64(samples as f64 / SAMPLE_RATE);
            samples += frame.pcm.len() as u64;
            let now = Duration::from_secs_f64(samples as f64 / SAMPLE_RATE);
            clock.observe(frame.t_capture, now);
            ring.push(&frame.pcm);

            let vad_events = match vad.push(&frame) {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("vad: {e:#}");
                    Vec::new()
                }
            };

            if let Some(fast) = p.fast.as_mut() {
                let evs = match rt.block_on(fast.feed(&frame.pcm, t_origin)) {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::warn!("fast lane: {e:#}");
                        Vec::new()
                    }
                };
                for ev in evs {
                    match ev {
                        AsrEvent::Partial { text, .. } => {
                            let text = punctuate(p.punct.as_mut(), &text);
                            emit(&p.out, &clock, &mut spec, stream.fast_partial(&text, now));
                        }
                        AsrEvent::Final {
                            words,
                            t_start,
                            t_end,
                            ..
                        } => {
                            let words = punctuate_words(p.punct.as_mut(), words);
                            emit(&p.out, &clock, &mut spec, stream.fast_final(&words, now));
                            // The accurate lane runs once over exactly this
                            // utterance: its boundaries come from the fast
                            // lane's own endpoint detector, which is the
                            // condition whisper is good at (PLAN §12.3).
                            offer(&p.seg_tx, &ring, t_start, t_end.min(now), &stream);
                        }
                    }
                }
            }

            // Accurate-only mode has no fast lane to borrow endpoints from, so
            // the VAD marks the utterances instead. Same rule either way: one
            // pass over one whole utterance.
            if p.mode == LaneMode::AccurateOnly {
                if vad_events.contains(&VadEvent::SpeechStart) && speech_start.is_none() {
                    speech_start = Some(now.saturating_sub(vad.silence()));
                }
                let ended = vad_events.iter().any(|e| matches!(e, VadEvent::SpeechEnd));
                if ended && vad.silence() >= pause_flush {
                    if let Some(a) = speech_start.take() {
                        offer(&p.seg_tx, &ring, a, now, &stream);
                    }
                } else if speech_start.is_some_and(|a| now.saturating_sub(a) >= max_segment) {
                    // A speaker who does not pause still has to reach the screen.
                    let a = speech_start.replace(now).expect("checked above");
                    offer(&p.seg_tx, &ring, a, now, &stream);
                }
            }

            while let Ok(words) = p.hyp_rx.try_recv() {
                emit(
                    &p.out,
                    &clock,
                    &mut spec,
                    stream.accurate_segment(&words, now),
                );
            }
            emit(&p.out, &clock, &mut spec, stream.tick(now));
            // The one place the fast lane's silence is spent on something. By
            // here the frame has been through both lanes, so if the words have
            // stopped moving they really have.
            if p.out.drafts {
                spec.tick(&p.out);
            }

            // How far behind the live audio this thread is, once it has done
            // everything one frame asks of it. Everything a reader feels is
            // downstream of this number, so it is the first one to look at --
            // but it starts at the capture callback, so the device's own
            // buffering is *not* in it. Once a second: at 31 frames a second a
            // per-frame line would itself cost latency.
            if lag_reported.elapsed() >= Duration::from_secs(1) {
                lag_reported = Instant::now();
                tracing::debug!(ms = clock.lag().as_millis() as u64, "pipeline lag");
            }
        }

        // Capture has ended. Flush what the engines are still holding, then let
        // the promotion rule finalise anything the accurate lane never answered
        // for, so the last thing anyone said still reaches the transcript.
        let end = Duration::from_secs_f64(samples as f64 / SAMPLE_RATE);
        if let Some(fast) = p.fast.as_mut()
            && let Ok(Some(AsrEvent::Final {
                words,
                t_start,
                t_end,
                ..
            })) = rt.block_on(fast.finalize())
        {
            let words = punctuate_words(p.punct.as_mut(), words);
            emit(&p.out, &clock, &mut spec, stream.fast_final(&words, end));
            offer(&p.seg_tx, &ring, t_start, t_end.min(end), &stream);
        }
        if let Some(n) = p
            .punct
            .as_ref()
            .map(OnlinePunct::mismatches)
            .filter(|n| *n > 0)
        {
            tracing::warn!("punctuation left {n} line(s) alone: the model changed the words");
        }
        drop(p.seg_tx);
        while let Some(words) = p.hyp_rx.blocking_recv() {
            emit(
                &p.out,
                &clock,
                &mut spec,
                stream.accurate_segment(&words, end),
            );
        }
        emit(
            &p.out,
            &clock,
            &mut spec,
            stream.tick(end + Duration::from_secs_f64(p.cfg.promote_after_s)),
        );
    }))
}

/// Punctuate one hypothesis, or hand it back untouched.
///
/// Both this and [`punctuate_words`] exist, and it is worth writing down why,
/// because "punctuate in one place" was the first plan. The fast lane's text
/// reaches the reader twice: while it is still being revised (`Partial`) and
/// once at the endpoint (`Final`). Punctuating only the second one would leave
/// the bar switching from shouting to sentences at every line boundary -- and
/// worse, it would break task 1.20. `Speculation` translates the *partial*
/// during the silence that ends a sentence and `Drafts::reuse` finds that
/// answer again by string equality with the closed line's text (29 of 29 hits
/// on `ami_meeting`). Punctuate one side only and every one of those becomes a
/// miss: a wasted NLLB pass, and Chinese 340 ms later than it is today.
///
/// Doing both is safe because the model is a deterministic text-to-text
/// function: the last partial before an endpoint says what the final words
/// say, so it punctuates to the same string and the match survives.
///
/// A failure returns the input. There is no state to corrupt and nothing to
/// retry -- this is the difference between shipping subtitles with no marks
/// and shipping none at all.
fn punctuate(punct: Option<&mut OnlinePunct>, text: &str) -> String {
    let Some(p) = punct else {
        return text.to_owned();
    };
    match p.restore(text) {
        Ok(out) => out,
        Err(e) => {
            tracing::warn!("punctuation: {e:#}");
            text.to_owned()
        }
    }
}

/// The same over timed words, which is the form the line boundaries are cut
/// from. `restore_words` carries every `start` and `end` through untouched and
/// refuses the whole line if the word count moved, so this cannot put a
/// boundary anywhere new -- content WER, drop rate and line count are provably
/// unchanged by it (task 1.25 S1).
fn punctuate_words(punct: Option<&mut OnlinePunct>, words: Vec<Word>) -> Vec<Word> {
    let Some(p) = punct else { return words };
    match p.restore_words(&words) {
        Ok(out) => out,
        Err(e) => {
            tracing::warn!("punctuation: {e:#}");
            words
        }
    }
}

/// Hand one utterance to the accurate lane, or give up on it.
///
/// Giving up is deliberate: the line it belongs to is then promoted from the
/// fast lane by the ordinary 8 s rule, which is worse text but still text and
/// still in the transcript. Queueing it instead would put the accurate lane
/// further behind on every utterance and never let it catch up.
fn offer(
    tx: &mpsc::Sender<Segment>,
    ring: &Ring,
    t_start: Duration,
    t_end: Duration,
    stream: &Stream,
) {
    let (pcm, t_origin) = ring.cut(t_start, t_end);
    if pcm.is_empty() {
        return;
    }
    let seg = Segment {
        pcm,
        t_origin,
        prompt: stream.prompt_tail(),
    };
    if let Err(mpsc::error::TrySendError::Full(_)) = tx.try_send(seg) {
        tracing::warn!(
            "the accurate lane is behind; {:.1}-{:.1}s will be promoted from the fast lane",
            t_start.as_secs_f64(),
            t_end.as_secs_f64()
        );
    }
}

/// Send finalised events on, and queue the translatable ones.
///
/// `blocking_send` for the sink: finalised text is never dropped (PLAN §11).
/// `try_send` for MT: a subtitle with no translation is a subtitle; a queue of
/// translations for lines that left the screen a minute ago is not.
///
/// Two kinds of line are queued. A settled one is the answer and reaches the
/// transcript. A draft is the fast lane's sentence the moment it closes, about
/// half a second before the accurate lane finalises the same line -- that
/// second is the whole reason it exists, and losing one costs nothing, so a
/// full queue drops it without a word. `SourcePartial { settled: false }` is
/// never queued: those arrive dozens of times a second.
fn emit(out: &Out, clock: &Clock, spec: &mut Speculation, events: Vec<EngineEvent>) {
    for ev in events {
        let mut metric = None;
        match &ev {
            EngineEvent::SourceFinal {
                line_id,
                text,
                t_end,
                lane,
                ..
            } => {
                spec.settle();
                let audio_end = clock.wall(*t_end);
                // The gate reading, at last: PLAN §16's G1 and G2 are both
                // "audio ends -> text on screen", and until now nothing in the
                // running program measured either. The eval harness did, over
                // files, which is not the same thing as a capture device and a
                // machine with other work to do.
                metric = Some(LatencySample {
                    line_id: *line_id,
                    lane: *lane,
                    stage: Stage::Source,
                    settled: true,
                    latency: audio_end.elapsed(),
                });
                if worth_translating(text) {
                    let job = MtJob {
                        line_id: *line_id,
                        text: text.clone(),
                        settled: true,
                        speculative: false,
                        audio_end,
                        lane: *lane,
                    };
                    if out.mt.try_send(job).is_err() {
                        tracing::warn!("translation queue full; line {line_id} stays in English");
                    }
                }
            }
            EngineEvent::SourcePartial {
                line_id,
                text,
                closed: None,
                ..
            } => {
                // Still moving. The guess waits for it to stop.
                spec.observe(*line_id, text);
            }
            EngineEvent::SourcePartial {
                line_id,
                text,
                closed: Some(t_end),
                ..
            } => {
                spec.settle();
                // The line's own span, the same origin the settled line will
                // use -- so the two readings for one line are the same
                // measurement taken twice, and subtracting them gives what the
                // fast lane bought.
                let audio_end = clock.wall(*t_end);
                // **This is G1** (PLAN §16): the fast lane's finished sentence
                // reaching the screen. The `SourceFinal` for the same line is
                // G2, and arrives about a second later. Reporting only the
                // second one would say the source row is a second slower than
                // a reader actually sees it change.
                metric = Some(LatencySample {
                    line_id: *line_id,
                    lane: Lane::Fast,
                    stage: Stage::Source,
                    settled: false,
                    latency: audio_end.elapsed(),
                });
                if out.drafts && worth_translating(text) {
                    let job = MtJob {
                        line_id: *line_id,
                        text: text.clone(),
                        settled: false,
                        speculative: false,
                        audio_end,
                        lane: Lane::Fast,
                    };
                    let _ = out.mt.try_send(job);
                }
            }
            _ => {}
        }
        if out.sink.blocking_send(SinkMsg::Event(ev)).is_err() {
            return;
        }
        if let Some(m) = metric
            && out
                .sink
                .blocking_send(SinkMsg::Event(EngineEvent::Metrics(m)))
                .is_err()
        {
            return;
        }
    }
}

/// A line of nothing but backchannel is not worth translating.
///
/// 20.5% of lines are `Okay.` / `Yeah.` / `Um...`, and NLLB turns every one of
/// them into a confident hallucination -- "沒有任何問題", "其他國家". Task 1.6
/// measured it; PLAN §19-20 is the entry. Dropping them here is half the fix
/// and costs nothing.
fn worth_translating(text: &str) -> bool {
    !li_stream::text::content_words(text).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use li_types::Lane;

    fn out(drafts: bool) -> (Out, mpsc::Receiver<SinkMsg>, mpsc::Receiver<MtJob>) {
        let (sink, sink_rx) = mpsc::channel(64);
        let (mt, mt_rx) = mpsc::channel(MT_QUEUE);
        (Out { sink, mt, drafts }, sink_rx, mt_rx)
    }

    fn partial(line_id: u64, text: &str, settled: bool) -> EngineEvent {
        EngineEvent::SourcePartial {
            line_id,
            text: text.into(),
            lane: Lane::Fast,
            closed: settled.then(|| Duration::from_secs(1)),
        }
    }

    fn queued(rx: &mut mpsc::Receiver<MtJob>) -> Vec<(u64, String, bool)> {
        let mut jobs = Vec::new();
        while let Ok(j) = rx.try_recv() {
            jobs.push((j.line_id, j.text, j.settled));
        }
        jobs
    }

    fn job(line_id: u64, settled: bool) -> MtJob {
        MtJob {
            line_id,
            text: "words enough to be worth a pass".into(),
            settled,
            speculative: false,
            audio_end: Instant::now(),
            lane: Lane::Fast,
        }
    }

    fn guess(line_id: u64) -> MtJob {
        MtJob {
            speculative: true,
            ..job(line_id, false)
        }
    }

    /// `(line_id, what kind)` for whatever the queue kept.
    fn survivors(jobs: impl IntoIterator<Item = MtJob>) -> Vec<(u64, &'static str)> {
        let mut q: VecDeque<MtJob> = jobs.into_iter().collect();
        drop_superseded_drafts(&mut q);
        q.iter()
            .map(|j| {
                (
                    j.line_id,
                    match (j.settled, j.speculative) {
                        (true, _) => "settled",
                        (_, true) => "guess",
                        _ => "draft",
                    },
                )
            })
            .collect()
    }

    #[test]
    fn a_draft_a_newer_line_has_overtaken_is_not_translated() {
        // Line 7's draft would reach the bar only to be replaced by line 8's,
        // and the pass it spent is 0.7 s that line 8 waited for nothing.
        assert_eq!(
            survivors([job(7, false), job(8, false)]),
            [(8, "draft")],
            "only the newest draft is still early"
        );
    }

    #[test]
    fn a_draft_is_dropped_once_its_own_settled_text_is_waiting() {
        // Same line, better words, already in the queue: the draft cannot be
        // early relative to the thing that replaces it.
        assert_eq!(survivors([job(7, false), job(7, true)]), [(7, "settled")]);
    }

    #[test]
    fn settled_lines_are_never_dropped_however_far_behind() {
        // They are what `on_translation_final` writes. PLAN §11: finalised
        // text does not disappear because the machine is busy.
        assert_eq!(
            survivors([job(1, true), job(2, true), job(3, true), job(4, false)]),
            [(1, "settled"), (2, "settled"), (3, "settled"), (4, "draft")]
        );
    }

    #[test]
    fn a_lone_draft_survives_because_nothing_has_overtaken_it() {
        assert_eq!(survivors([job(5, false)]), [(5, "draft")]);
    }

    // Plain `#[test]`: `emit` blocks on the sink, which a runtime thread may
    // not do.
    #[test]
    fn a_revisable_partial_is_never_translated_but_a_settled_one_is() {
        // The fast lane emits partials dozens of times a second. Queueing one
        // would spend an NLLB pass on text that is about to be replaced, and
        // fill the queue so the line that mattered was dropped.
        let (o, _sink, mut mt) = out(true);
        emit(
            &o,
            &Clock::default(),
            &mut Speculation::default(),
            vec![
                partial(1, "so lets get", false),
                partial(1, "so lets get started", false),
                partial(1, "so lets get started with the agenda", true),
            ],
        );
        assert_eq!(
            queued(&mut mt),
            [(1, "so lets get started with the agenda".to_string(), false)]
        );
    }

    #[test]
    fn drafting_off_leaves_exactly_the_old_one_translation_per_line() {
        let (o, _sink, mut mt) = out(false);
        emit(
            &o,
            &Clock::default(),
            &mut Speculation::default(),
            vec![partial(1, "so lets get started", true)],
        );
        assert!(queued(&mut mt).is_empty());
    }

    #[test]
    fn a_promoted_line_reuses_its_draft_instead_of_translating_it_twice() {
        // The accurate lane timed out, so the line is settled with the fast
        // lane's own words -- the very string the draft was made from.
        let mut d = Drafts::default();
        d.remember(7, "the demo is on friday".into(), "展示在週五");
        assert_eq!(
            d.reuse(7, "the demo is on friday", true).as_deref(),
            Some("展示在週五")
        );
    }

    #[test]
    fn a_line_the_accurate_lane_answered_is_translated_again() {
        let mut d = Drafts::default();
        d.remember(7, "the demo is on friday".into(), "展示在週五");
        // Different words: punctuation, capitals, and whatever whisper heard
        // differently. Reusing here would put the fast lane's mistakes in the
        // transcript under the accurate lane's name.
        assert_eq!(d.reuse(7, "The demo is on Friday.", true), None);
        // ...and the entry is gone either way, so the next line's draft cannot
        // be matched against a settled line that already went past.
        assert_eq!(d.reuse(7, "the demo is on friday", true), None);
    }

    #[test]
    fn a_draft_takes_the_guess_made_for_its_own_line_and_nobody_elses() {
        // This is what makes speculating pay: the line closed saying exactly
        // what was guessed, so the draft costs nothing at all.
        let mut d = Drafts::default();
        d.remember(7, "same words".into(), "一樣的字");
        assert_eq!(d.reuse(8, "same words", false), None, "another line");
        assert_eq!(d.reuse(7, "other words", false), None, "other words");
        assert_eq!(d.reuse(7, "same words", false).as_deref(), Some("一樣的字"));
    }

    #[test]
    fn a_guess_the_line_did_not_confirm_is_dropped_when_the_line_settles() {
        // The fast lane paused mid-sentence, the guess translated half of it,
        // and the sentence then went somewhere else. Nothing must be able to
        // pick that answer up later under a line it does not belong to.
        let mut d = Drafts::default();
        d.remember(7, "the demo is".into(), "展示是");
        assert_eq!(d.reuse(7, "The demo is on Friday.", true), None);
        assert_eq!(
            d.reuse(7, "the demo is", false),
            None,
            "cleared with the line"
        );
    }

    /// Push the guess's clock back so the threshold is already past, without
    /// a test that sleeps for a third of a second.
    fn ready(spec: &mut Speculation) {
        let g = spec.0.as_mut().expect("something to guess at");
        g.since -= SPECULATE_AFTER;
    }

    #[test]
    fn a_line_still_being_revised_restarts_the_clock_but_a_repeat_does_not() {
        let (o, _sink, mut mt) = out(true);
        let mut spec = Speculation::default();
        spec.observe(1, "so lets get");
        ready(&mut spec);
        // sherpa re-sends the same hypothesis: the words have not moved, so
        // this must not push the guess back out of reach.
        spec.observe(1, "so lets get");
        spec.tick(&o);
        assert_eq!(queued(&mut mt).len(), 1, "the words stopped moving");

        spec.observe(1, "so lets get started");
        spec.tick(&o);
        assert!(queued(&mut mt).is_empty(), "moved again; wait again");
    }

    #[test]
    fn a_guess_is_sent_once_and_never_after_the_line_closes() {
        let (o, _sink, mut mt) = out(true);
        let mut spec = Speculation::default();
        spec.observe(1, "so lets get started with the agenda");
        ready(&mut spec);
        spec.tick(&o);
        spec.tick(&o);
        assert_eq!(queued(&mut mt).len(), 1, "once, not once per frame");

        // The line closed. `emit` has queued the real draft; there is nothing
        // left to be early about.
        spec.settle();
        spec.tick(&o);
        assert!(queued(&mut mt).is_empty());
    }

    #[test]
    fn backchannel_is_not_worth_guessing_at_either() {
        let (o, _sink, mut mt) = out(true);
        let mut spec = Speculation::default();
        spec.observe(1, "okay yeah");
        ready(&mut spec);
        spec.tick(&o);
        assert!(queued(&mut mt).is_empty());
    }

    #[test]
    fn a_guess_is_the_first_thing_the_queue_gives_up() {
        // Same line, and the real draft for it is already waiting: the guess
        // cannot be early any more, and the pass it would spend is 0.3-0.9 s
        // the draft behind it waits for nothing.
        assert_eq!(survivors([guess(7), job(7, false)]), [(7, "draft")]);
        assert_eq!(survivors([guess(7), guess(8)]), [(8, "guess")]);
        assert_eq!(survivors([guess(7), job(7, true)]), [(7, "settled")]);
        assert_eq!(survivors([guess(7)]), [(7, "guess")], "nothing overtook it");
    }

    #[test]
    fn a_line_of_backchannel_is_not_sent_to_the_translator() {
        assert!(!worth_translating("Okay."));
        assert!(!worth_translating("Yeah, um, okay."));
        assert!(!worth_translating("   "));
        assert!(worth_translating("Okay, this is our agenda."));
        assert!(worth_translating("Hello everybody."));
    }

    #[test]
    fn both_lanes_off_is_a_startup_error_naming_the_sections() {
        let mut cfg = EngineConfig::default();
        cfg.asr.fast = None;
        cfg.asr.accurate = None;
        let err = lane_mode(&cfg).unwrap_err().to_string();
        assert!(err.contains("[asr.fast]"), "{err}");
    }

    #[test]
    fn switching_a_lane_off_picks_the_single_lane_mode() {
        let mut cfg = EngineConfig::default();
        assert_eq!(lane_mode(&cfg).unwrap(), LaneMode::Dual);
        cfg.asr.accurate.as_mut().unwrap().backend = "off".into();
        assert_eq!(lane_mode(&cfg).unwrap(), LaneMode::FastOnly);
        cfg.asr.fast = None;
        cfg.asr.accurate.as_mut().unwrap().backend = "whispercpp".into();
        assert_eq!(lane_mode(&cfg).unwrap(), LaneMode::AccurateOnly);
    }
}
