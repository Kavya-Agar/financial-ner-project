// Builds a flat list of contiguous, non-overlapping text segments from the
// original text plus a (possibly malformed) list of entities. Guaranteed
// invariants, even with bad input:
//   - segments are in ascending order with no gaps and no overlaps
//   - concatenating every segment's `text` reproduces the original `text`
//     exactly
//   - never throws: out-of-range offsets are clamped, reversed/zero-length
//     ranges are dropped, and overlapping entities are resolved by giving
//     priority to whichever entity starts first (later, overlapping
//     entities are clipped or dropped rather than causing a crash)
export function buildSegments(text, entities) {
  const safeText = typeof text === 'string' ? text : '';
  const length = safeText.length;

  const valid = (Array.isArray(entities) ? entities : [])
    .filter(
      (e) =>
        e &&
        typeof e.start === 'number' &&
        typeof e.end === 'number' &&
        typeof e.label === 'string' &&
        Number.isFinite(e.start) &&
        Number.isFinite(e.end),
    )
    .map((e) => ({
      ...e,
      start: Math.max(0, Math.min(Math.floor(e.start), length)),
      end: Math.max(0, Math.min(Math.floor(e.end), length)),
    }))
    .filter((e) => e.end > e.start)
    // Sort by start ascending, then by longest-first so a larger entity
    // "wins" over a shorter one that starts at the same offset.
    .sort((a, b) => a.start - b.start || b.end - a.end);

  const segments = [];
  let cursor = 0;

  for (const entity of valid) {
    if (entity.end <= cursor) {
      // Entirely consumed by a previously emitted (overlapping) entity.
      continue;
    }
    const start = Math.max(entity.start, cursor);
    if (start > cursor) {
      segments.push({ text: safeText.slice(cursor, start), label: null });
    }
    segments.push({
      text: safeText.slice(start, entity.end),
      label: entity.label,
      score: entity.score,
    });
    cursor = entity.end;
  }

  if (cursor < length) {
    segments.push({ text: safeText.slice(cursor), label: null });
  }

  return segments;
}

export default function HighlightedText({ text, entities }) {
  const segments = buildSegments(text, entities);

  return (
    <div className="highlighted-text" data-testid="highlighted-text">
      {segments.map((segment, index) =>
        segment.label ? (
          <mark
            key={index}
            className={`entity-highlight entity-${segment.label}`}
            data-label={segment.label}
            data-testid="entity-span"
            title={
              typeof segment.score === 'number'
                ? `${segment.label} (${segment.score.toFixed(2)})`
                : segment.label
            }
          >
            {segment.text}
          </mark>
        ) : (
          <span key={index} className="plain-text" data-testid="plain-span">
            {segment.text}
          </span>
        ),
      )}
    </div>
  );
}
