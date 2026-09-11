# Golden eval assets

Each clip is `<name>.wav` (16 kHz mono s16) plus references:
`<name>.en.txt` (ASR reference) and `<name>.zh.txt` (MT reference).

| clip | length | source | character |
|---|---|---|---|
| `jfk` | 11s | JFK inaugural excerpt | too short for stable statistics — smoke test only |
| `jfk_x4` | 48s | **synthetic** (`jfk` repeated ×4) | ⚠️ **excluded from evals** — verbatim repetition provokes Whisper hallucination, so its WER is meaningless |
| `read_clean` | 82s | LibriSpeech `dev-clean` 3853/163249 | clean read speech, female |
| `read_clean2` | 84s | LibriSpeech `dev-clean` 3000/15664 | clean read speech |
| `read_hard` | 82s | LibriSpeech `dev-other` 3663/172528 | harder acoustics / accent |
| `ami_meeting` | 100s | AMI `ES2004a`, single distant mic | **the G2 sign-off clip** — 4-person meeting, far-field, human reference |
| `ami_meeting2` | 100s | AMI `EN2002a`, single distant mic | same, non-scenario (real) meeting |

## Provenance

- **Audio + English transcripts**: [LibriSpeech](https://www.openslr.org/12/) (CC BY 4.0),
  built from consecutive utterances from one chapter
  concatenated with 0.7s gaps, so the fast lane's endpoint detector has natural
  sentence breaks to fire on. English references are LibriSpeech's own verified
  transcripts (upper-case, no punctuation).
- **Chinese references**: written by Claude, **not human-verified**. They are good
  enough for *relative* comparison (fast lane vs accurate lane through the same MT
  engine), which is what the evals need. Do not quote the absolute chrF as a
  quality claim; a human-translated reference set is still owed before MVP sign-off.

- **AMI clips**: [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
  (**CC-BY-4.0**, University of Edinburgh), cut
  from the corpus `test` split. Audio is `Array1-01` — a **single distant microphone**,
  the same far-field condition as a laptop on a meeting table, and the reason these
  are the sign-off clips rather than the `ihm` headset channels.
  The reference is AMI's own human word-level annotation, verbatim: filler words,
  false starts and truncations are all in it (`OKAY TU TU TU TU HELLO EVERYBODY...`).
  Whisper is trained to *omit* filler, so `cargo xtask eval` reports
  WER both raw and with filler stripped from both sides.

  **Overlapped speech is capped at 10%** (`--max-overlap`). One microphone can only
  carry one voice, so a window full of cross-talk measures AMI's overlap rate rather
  than the model — the first window picked was 21.9% overlap and was rejected. The
  two committed clips sit at 2.1% and 2.9%, which also means they are **not**
  worst-case AMI; a real meeting with heavy cross-talk will score worse.

## Private clips (`testdata/private/`, git-ignored)

Recordings of real meetings, kept on the developer's machine as a smoke test on
real microphones and real rooms. **Never committed** — they have identifiable
speakers — and neither is anything made from them: a transcript or an eval
report of a private recording carries the same words. Report *numbers* from
these clips, never their text.

## Fast speech: made, not recorded

Nothing here is fast enough to reach `max_utterance_s` (12 s) — so the failure a
user reported, a line running the full 20 s and the accurate lane timing out on
it, cannot be reproduced from these files.

Speed one up instead. `atempo` changes rate without changing pitch, so the
reference transcript stays valid and WER still means something:

```
ffmpeg -i testdata/read_clean.wav -filter:a "atempo=1.5" -ar 16000 -ac 1 fast_1.5.wav
cargo run -q --release -p xtask -- eval --wav fast_1.5.wav --ref testdata/read_clean.en.txt
```

1.5x is where it breaks: the longest utterance goes 5.4 s -> 20.0 s (90 words)
and the accurate lane gives up on it. 1.3x only reaches 8.2 s, which is still
inside the cap. The file is not committed -- it is one command away, and 1.7 MB
of derived audio in git is 1.7 MB of audio nobody will re-derive when the
recipe changes.

## Coverage gap

| condition | covered by | reference quality |
|---|---|---|
| clean read speech | `read_clean`, `read_clean2` | **gold** (LibriSpeech), but **in-domain** for the fast lane's model → optimistic |
| harder read speech | `read_hard` | **gold**, in-domain |
| far-field multi-party meeting | `ami_meeting`, `ami_meeting2` | **gold** (human), out-of-domain → this is what G2 is signed off on |
| real microphones and rooms | `private/` (not committed) | smoke test only |

Still missing: **non-native and regional accents** (AMI is 2000s British/European
academic English), **background noise / music**, and a **human-translated zh-TW
reference** for gate G5 — every `.zh.txt` here was written by Claude and is
explicitly not human-verified, so absolute chrF is not quotable.
