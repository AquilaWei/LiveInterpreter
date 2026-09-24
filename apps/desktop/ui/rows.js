// Which translations are on the bar.
//
// The bar used to show one translation, and a new one replaced it the moment
// it arrived. That is fine at a talking pace and not for a fast speaker: over
// two sessions on 2026-09-24, 13 of 67 translations were replaced before they
// could have been read at 7 characters a second -- a 63-character sentence was
// on screen for 0.4 s. Holding each one for a reading time instead makes the
// bar fall behind the speaker and then skip lines to catch up.
//
// So the last `keep` translations stay: a sentence stays readable for as long
// as the next one takes to arrive *and* the one after that. Replayed over the same
// two sessions, keeping two left 4 of 67 unreadable, with no delay added.
//
// Kept apart from `main.js` so `node --test` can check it without a window.

// Where `lineId` goes, given the ids now on the bar (oldest first).
//
// Returns the new list, oldest first; main.js paints it the other way up, so
// the newest sits right under the English it translates. A line already on the bar is updated where it is --
// that is the accurate lane's settled translation replacing the draft, and it
// can now land on a line the bar has moved past as long as it is still in
// view. A line older than the newest one shown, and not among them, is
// dropped: the rows never go backwards, for the same reason the source row
// does not (see main.js).
export function place(recent, lineId, keep) {
  if (recent.includes(lineId)) return recent;
  const newest = recent.length ? recent[recent.length - 1] : -Infinity;
  if (lineId < newest) return recent;
  return [...recent, lineId].slice(-Math.max(1, keep));
}
