//! A recorded minute of subtitle, for looking at the bar without an engine.
//!
//! This asks a question no measurement answers: when the accurate lane
//! replaces the fast lane's text in place, is reading disturbed? Only a person
//! can say, and asking them costs 3 GB of models, a machine with a GPU, and
//! something in English playing -- so it never gets asked, and the answer
//! arrives after the UI is written.
//!
//! `LI_PROBE=replay` emits real [`UiEvent`]s through the real channel on
//! measured timings, so what is on screen is the production render path fed
//! from a script instead of from audio. The script is built around the three
//! cases that are awkward on a two-line bar, and each is there on purpose:
//!
//! 1. **The overwrite.** A pause after the sentence, so the accurate lane's
//!    punctuated text lands while its own line is still on the bar. The good
//!    case, and the one the commit policy is written for.
//! 2. **The late answer.** Continuous speech, so the accurate lane's answer for
//!    a line arrives after the next line has already taken the row. The bar
//!    drops it (see `main.js`): the transcript keeps it, and the row never
//!    walks backwards.
//! 3. **The two clocks.** A translation for the previous sentence sitting under
//!    the current one -- unavoidable when text appears at 0.8 s and its
//!    translation at 4 s, and the thing most likely to read as a bug.

use std::time::Duration;

use li_types::{EngineStatus, Lane, UiEvent};
use tauri::Emitter;

/// Lines in one pass of the script. Ids carry on across repeats, because the
/// bar ignores a line older than the one it is showing -- restarting at 1 would
/// freeze it after the first loop.
const LINES: u64 = 5;

/// `(milliseconds from the start of the loop, event)`.
///
/// The offsets are the measured ones: the fast lane reaches the screen ~0.8 s
/// after the audio it describes, the accurate lane ~2 s after that,
/// and the translation ~0.3 s after the text it is made from.
fn script(base: u64) -> Vec<(u64, UiEvent)> {
    let id = |n: u64| base + n;
    let partial = |n: u64, text: &str| UiEvent::Partial {
        line_id: id(n),
        text: text.to_owned(),
        lane: Lane::Fast,
        settled: false,
    };
    // The fast lane's endpoint: the same event, but the words have stopped
    // moving. It is what the draft translation is made from.
    let closed = |n: u64, text: &str| UiEvent::Partial {
        line_id: id(n),
        text: text.to_owned(),
        lane: Lane::Fast,
        settled: true,
    };
    let settled = |n: u64, text: &str, at: f64, lane: Lane| UiEvent::Final {
        line_id: id(n),
        text: text.to_owned(),
        start_s: at,
        end_s: at + 3.0,
        lane,
        reason: None,
    };
    let zh = |n: u64, text: &str| UiEvent::Translation {
        line_id: id(n),
        text: text.to_owned(),
        settled: true,
    };
    // Translated from fast-lane text, so: no punctuation upstream, and it is
    // replaced under the same id a second or so later.
    let draft = |n: u64, text: &str| UiEvent::Translation {
        line_id: id(n),
        text: text.to_owned(),
        settled: false,
    };

    vec![
        // 1. The speaker pauses, so the overwrite happens in full view.
        (600, partial(1, "so let's get")),
        (1100, partial(1, "so let's get started with")),
        (1600, partial(1, "so let's get started with the agenda")),
        (
            2100,
            partial(1, "so let's get started with the agenda for today"),
        ),
        // The fast lane hits its endpoint here. The draft translation follows
        // it by one NLLB pass, which is what buys the half second: without it
        // the Chinese row stays empty until 4300.
        (
            2400,
            closed(1, "so let's get started with the agenda for today"),
        ),
        (2600, draft(1, "我們今天就從議程開始吧")),
        (
            4000,
            settled(
                1,
                "So let's get started with the agenda for today.",
                1.3,
                Lane::Accurate,
            ),
        ),
        (4300, zh(1, "那我們就從今天的議程開始吧。")),
        // 2. Now they do not pause. Line 3 opens before line 2 is settled, so
        //    line 2's accurate text arrives too late for the screen.
        (5800, partial(2, "the first item is the release schedule")),
        (
            6300,
            partial(2, "the first item is the release schedule and i think"),
        ),
        (
            6800,
            partial(
                2,
                "the first item is the release schedule and i think we should move it by two weeks",
            ),
        ),
        (7600, partial(3, "the second item is the localisation work")),
        (
            8100,
            settled(
                2,
                "The first item is the release schedule, and I think we should move it by two weeks.",
                5.0,
                Lane::Accurate,
            ),
        ),
        // 3. ...and its translation arrives under line 3's text.
        (8500, zh(2, "第一項是發布時程，我認為我們應該延後兩週。")),
        (
            9000,
            partial(
                3,
                "the second item is the localisation work which is nearly done for traditional chinese",
            ),
        ),
        (
            10600,
            settled(
                3,
                "The second item is the localisation work, which is nearly done for Traditional Chinese.",
                7.0,
                Lane::Accurate,
            ),
        ),
        (11000, zh(3, "第二項是在地化工作，繁體中文的部分快完成了。")),
        // 4. A line the accurate lane never answered: promoted fast-lane text,
        //    with no punctuation and no capitals. This is what 8 s of silence
        //    from whisper looks like on the bar.
        (
            12600,
            partial(4, "and the last thing is the demo on friday"),
        ),
        (12900, closed(4, "and the last thing is the demo on friday")),
        (13100, draft(4, "最後一件事是週五的展示")),
        (
            13400,
            settled(
                4,
                "and the last thing is the demo on friday",
                11.0,
                Lane::Fast,
            ),
        ),
        (13800, zh(4, "最後一件事是週五的展示。")),
        // 5. A line at the 40-word cap, with a translation that needs both of
        //    its rows: the two cases where the row budgets actually bite.
        (
            15200,
            partial(
                5,
                "and if we are going to ship the release in two weeks then the documentation and the localisation and the release notes all have to be finished before the end of next week which is going to be tight",
            ),
        ),
        (
            17000,
            settled(
                5,
                "And if we are going to ship the release in two weeks, then the documentation and the localisation and the release notes all have to be finished before the end of next week, which is going to be tight.",
                13.0,
                Lane::Accurate,
            ),
        ),
        (
            17400,
            zh(
                5,
                "如果我們要在兩週內發布，那麼文件、在地化以及發布說明都必須在下週結束前完成，時間會相當緊迫。",
            ),
        ),
    ]
}

/// Play the script on a loop until the window closes.
pub fn spawn(app: tauri::AppHandle) {
    tauri::async_runtime::spawn(async move {
        let _ = app.emit(
            "engine://event",
            UiEvent::Status {
                status: EngineStatus::ModelLoading {
                    what: "nothing — this is a replay".into(),
                },
                text: "replay".into(),
            },
        );
        for pass in 0.. {
            let mut at = 0u64;
            for (t, ev) in script(pass * LINES) {
                tokio::time::sleep(Duration::from_millis(t - at)).await;
                at = t;
                if app.emit("engine://event", ev).is_err() {
                    return;
                }
            }
            // Long enough to see the bar at rest, holding the last line.
            tokio::time::sleep(Duration::from_millis(2500)).await;
        }
    });
}
