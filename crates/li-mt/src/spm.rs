//! A SentencePiece `.spm` model, read into a `tokenizers::Tokenizer`.
//!
//! Opus-MT ships its tokenizers as SentencePiece models and nothing else. The
//! obvious way to read them, the `sentencepiece` crate, builds Google's C++
//! library -- which carries its own copy of protobuf, and so does the ONNX
//! Runtime that `li-vad` links statically: the two collide at link time, twenty
//! duplicate symbols (tried 2026-09-24). So the model file is read here and
//! rebuilt from the pieces `tokenizers` already has, the same conversion
//! Hugging Face's `SpmConverter` does: a Unigram model over the file's own
//! pieces and scores, its precompiled NFKC normaliser, and the `▁`
//! word-boundary pre-tokenizer.
//!
//! Only what a Unigram model needs is read. A BPE `.spm` is refused rather
//! than tokenized wrongly.
//!
//! ## The file format
//!
//! A protobuf `ModelProto` (sentencepiece_model.proto). The three fields used:
//!
//! | field | what | used for |
//! |---|---|---|
//! | 1 `pieces` | repeated `{1 piece, 2 score, 3 type}` | the vocabulary |
//! | 2 `trainer_spec` | `3 model_type` | refusing anything but Unigram |
//! | 3 `normalizer_spec` | `2 precompiled_charsmap`, `3 add_dummy_prefix`, `4 remove_extra_whitespaces` | the normaliser |

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use tokenizers::{
    NormalizerWrapper, Tokenizer,
    models::unigram::Unigram,
    normalizers::{Precompiled, Replace, Sequence, Strip, replace::ReplacePattern},
    pre_tokenizers::metaspace::{Metaspace, PrependScheme},
};

/// sentencepiece_model.proto `ModelType::UNIGRAM`, also the default when the
/// field is absent.
const UNIGRAM: u64 = 1;
/// sentencepiece_model.proto `SentencePiece::Type::UNKNOWN`.
const UNKNOWN: u64 = 2;

/// Read `path` and build the tokenizer it describes. Fails on a file that is
/// not a Unigram SentencePiece model.
pub fn unigram(path: &Path) -> Result<Tokenizer> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let model = Model::parse(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    if model.model_type != UNIGRAM {
        bail!(
            "{} is a SentencePiece model of type {}, and only Unigram is supported",
            path.display(),
            model.model_type
        );
    }
    let unk = model.pieces.iter().position(|p| p.kind == UNKNOWN);
    let vocab = model
        .pieces
        .into_iter()
        .map(|p| (p.piece, p.score))
        .collect();
    let mut tok = Tokenizer::new(Unigram::from(vocab, unk, false).map_err(|e| anyhow!("{e}"))?);

    let mut norm: Vec<NormalizerWrapper> = Vec::new();
    if !model.charsmap.is_empty() {
        let pre = Precompiled::from(&model.charsmap).map_err(|e| anyhow!("{e}"))?;
        norm.push(pre.into());
    }
    if model.remove_extra_whitespaces {
        // SentencePiece's own rule: trim both ends, then collapse the runs.
        norm.push(Strip::new(true, true).into());
        let collapse =
            Replace::new(ReplacePattern::Regex(" {2,}".into()), " ").map_err(|e| anyhow!("{e}"))?;
        norm.push(collapse.into());
    }
    tok.with_normalizer(Some(Sequence::new(norm)))
        .map_err(|e| anyhow!("{e}"))?;
    let prepend = if model.add_dummy_prefix {
        PrependScheme::Always
    } else {
        PrependScheme::Never
    };
    tok.with_pre_tokenizer(Some(Metaspace::new('▁', prepend, true)));
    Ok(tok)
}

struct Piece {
    piece: String,
    score: f64,
    kind: u64,
}

struct Model {
    pieces: Vec<Piece>,
    model_type: u64,
    charsmap: Vec<u8>,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
}

impl Model {
    fn parse(bytes: &[u8]) -> Result<Self> {
        let mut m = Model {
            pieces: Vec::new(),
            model_type: UNIGRAM,
            charsmap: Vec::new(),
            // Both default to true in the .proto.
            add_dummy_prefix: true,
            remove_extra_whitespaces: true,
        };
        for field in Fields(bytes) {
            match field? {
                (1, Value::Bytes(b)) => m.pieces.push(Piece::parse(b)?),
                (2, Value::Bytes(b)) => {
                    for f in Fields(b) {
                        if let (3, Value::Varint(v)) = f? {
                            m.model_type = v;
                        }
                    }
                }
                (3, Value::Bytes(b)) => {
                    for f in Fields(b) {
                        match f? {
                            (2, Value::Bytes(c)) => m.charsmap = c.to_vec(),
                            (3, Value::Varint(v)) => m.add_dummy_prefix = v != 0,
                            (4, Value::Varint(v)) => m.remove_extra_whitespaces = v != 0,
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        if m.pieces.is_empty() {
            bail!("no pieces: not a SentencePiece model");
        }
        Ok(m)
    }
}

impl Piece {
    fn parse(bytes: &[u8]) -> Result<Self> {
        // `type` defaults to NORMAL (1).
        let (mut piece, mut score, mut kind) = (None, 0.0, 1);
        for f in Fields(bytes) {
            match f? {
                (1, Value::Bytes(b)) => piece = Some(String::from_utf8(b.to_vec())?),
                (2, Value::Fixed32(v)) => score = f64::from(f32::from_bits(v)),
                (3, Value::Varint(v)) => kind = v,
                _ => {}
            }
        }
        Ok(Piece {
            piece: piece.ok_or_else(|| anyhow!("a piece with no text"))?,
            score,
            kind,
        })
    }
}

enum Value<'a> {
    Varint(u64),
    Fixed32(u32),
    Bytes(&'a [u8]),
    Fixed64,
}

/// The fields of one protobuf message, in order: `(field number, value)`.
struct Fields<'a>(&'a [u8]);

impl<'a> Iterator for Fields<'a> {
    type Item = Result<(u64, Value<'a>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0.is_empty() {
            return None;
        }
        Some(self.field())
    }
}

impl<'a> Fields<'a> {
    fn field(&mut self) -> Result<(u64, Value<'a>)> {
        let key = self.varint()?;
        let value = match key & 7 {
            0 => Value::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Value::Fixed64
            }
            2 => {
                let len = usize::try_from(self.varint()?)?;
                Value::Bytes(self.take(len)?)
            }
            5 => Value::Fixed32(u32::from_le_bytes(self.take(4)?.try_into()?)),
            w => bail!("protobuf wire type {w} is not used by SentencePiece"),
        };
        Ok((key >> 3, value))
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *self.take(1)?.first().unwrap_or(&0);
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        bail!("protobuf varint longer than 64 bits")
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.0.len() < n {
            bail!("protobuf message cut short");
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-piece model: `<unk>` (type 2) and `▁a` with score -1.5, no
    /// normaliser. Written by hand from the .proto so the parser is checked
    /// without a model file.
    const TINY: &[u8] = &[
        0x0a, 0x09, 0x0a, 0x05, b'<', b'u', b'n', b'k', b'>', 0x18, 0x02, // piece 1
        0x0a, 0x0b, 0x0a, 0x04, 0xe2, 0x96, 0x81, b'a', 0x15, 0x00, 0x00, 0xc0,
        0xbf, // piece 2
    ];

    #[test]
    fn the_pieces_and_their_scores_are_read() {
        let m = Model::parse(TINY).unwrap();
        let got: Vec<(&str, f64, u64)> = m
            .pieces
            .iter()
            .map(|p| (p.piece.as_str(), p.score, p.kind))
            .collect();
        assert_eq!(got, [("<unk>", 0.0, 2), ("▁a", -1.5, 1)]);
    }

    #[test]
    fn a_model_with_no_trainer_spec_is_taken_as_unigram() {
        assert_eq!(Model::parse(TINY).unwrap().model_type, UNIGRAM);
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_panic() {
        assert!(Model::parse(&TINY[..TINY.len() - 2]).is_err());
    }

    #[test]
    fn a_file_with_no_pieces_is_not_a_model() {
        assert!(Model::parse(&[]).is_err());
    }
}
