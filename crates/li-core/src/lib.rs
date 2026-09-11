//! Pipeline assembly, configuration and the event bus (PLAN §10.3, §11).
//!
//! Every other crate is a piece with no opinion about the others; this one wires
//! them together and owns the threading rules of §11 — the capture callback only
//! copies into a ring buffer, inference runs on blocking threads, and the
//! channels between stages are bounded so a slow stage drops old audio rather
//! than growing an unbounded queue.

pub mod config;
pub mod engine;
pub mod models;

pub use engine::Engine;
