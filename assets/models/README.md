# Bundled models

Only the VAD lives here. Every other model is downloaded on first use into
`~/.cache/liveinterpreter/models/` and verified against `assets/models.toml`
(PLAN §15) — the recognisers and the translator are hundreds of megabytes and
have no business in a git repository or an installer.

| file | model | licence | sha256 |
|---|---|---|---|
| `silero_vad.onnx` | [silero-vad](https://github.com/snakers4/silero-vad) v5, `src/silero_vad/data/silero_vad.onnx` | MIT | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |

`li-vad` embeds this file with `include_bytes!`. At 2.2 MB that is a fair
trade: the VAD decides whether the recognisers run at all, so "the model is
missing" is a failure mode worth designing out rather than handling.

**Do not substitute faster-whisper's `silero_vad_v6.onnx`.** It is a repackaged
export with a different signature — `[seq_len, 576]` input and separate `h`/`c`
state tensors, scoring a batch of windows per call — and swapping the file
without changing the code gives wrong answers, not a load error.
