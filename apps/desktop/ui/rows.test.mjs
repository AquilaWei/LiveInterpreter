// `node --test apps/desktop/ui/*.test.mjs`
import { test } from "node:test";
import assert from "node:assert/strict";

import { place } from "./rows.js";

test("a new translation is added after the one already there", () => {
  assert.deepEqual(place([4], 5, 2), [4, 5]);
});

test("the oldest translation leaves when the bar is full", () => {
  assert.deepEqual(place([4, 5], 6, 2), [5, 6]);
});

test("keeping one is the old single-row bar", () => {
  assert.deepEqual(place([4], 5, 1), [5]);
});

test("a settled translation of a line still on the bar stays where it is", () => {
  assert.deepEqual(place([4, 5], 4, 2), [4, 5]);
});

test("a translation older than the newest one shown is dropped", () => {
  assert.deepEqual(place([4, 6], 5, 2), [4, 6]);
});

test("the first translation starts the bar", () => {
  assert.deepEqual(place([], 1, 2), [1]);
});

test("lines skipped by the recogniser leave no gap", () => {
  assert.deepEqual(place([4, 5], 9, 2), [5, 9]);
});
