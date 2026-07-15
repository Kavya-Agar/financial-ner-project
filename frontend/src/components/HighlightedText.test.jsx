import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import HighlightedText, { buildSegments } from './HighlightedText.jsx';

function reconstruct(segments) {
  return segments.map((s) => s.text).join('');
}

describe('buildSegments', () => {
  it('reproduces the original text exactly with a normal, well-formed set of entities', () => {
    const text = 'John Smith paid $500 to Acme Corp on 2024-01-05.';
    const entities = [
      { text: 'John Smith', label: 'PER', start: 0, end: 10, score: 0.98 },
      { text: '$500', label: 'AMOUNT', start: 16, end: 20, score: 0.9 },
      { text: 'Acme Corp', label: 'ORG', start: 24, end: 33, score: 0.95 },
      { text: '2024-01-05', label: 'DATE', start: 37, end: 47, score: 0.99 },
    ];
    const segments = buildSegments(text, entities);
    expect(reconstruct(segments)).toBe(text);
    const labeled = segments.filter((s) => s.label);
    expect(labeled).toHaveLength(4);
    expect(labeled.map((s) => s.label)).toEqual(['PER', 'AMOUNT', 'ORG', 'DATE']);
  });

  it('handles an entity at the very start of the text', () => {
    const text = 'Acme Corp is a company.';
    const entities = [{ label: 'ORG', start: 0, end: 9 }];
    const segments = buildSegments(text, entities);
    expect(reconstruct(segments)).toBe(text);
    expect(segments[0]).toMatchObject({ label: 'ORG', text: 'Acme Corp' });
  });

  it('handles an entity at the very end of the text', () => {
    const text = 'The company is Acme Corp';
    const entities = [{ label: 'ORG', start: 15, end: 25 }];
    const segments = buildSegments(text, entities);
    expect(reconstruct(segments)).toBe(text);
    expect(segments[segments.length - 1]).toMatchObject({
      label: 'ORG',
      text: 'Acme Corp',
    });
  });

  it('handles adjacent entities with zero gap between them', () => {
    const text = 'JohnSmith';
    const entities = [
      { label: 'PER', start: 0, end: 4 }, // "John"
      { label: 'PER', start: 4, end: 9 }, // "Smith"
    ];
    const segments = buildSegments(text, entities);
    expect(reconstruct(segments)).toBe(text);
    // No empty plain-text segment should be inserted between them.
    expect(segments).toHaveLength(2);
    expect(segments[0]).toMatchObject({ text: 'John', label: 'PER' });
    expect(segments[1]).toMatchObject({ text: 'Smith', label: 'PER' });
  });

  it('handles an entity spanning the entire text (no leading/trailing plain segments)', () => {
    const text = 'Acme Corp';
    const entities = [{ label: 'ORG', start: 0, end: 9 }];
    const segments = buildSegments(text, entities);
    expect(reconstruct(segments)).toBe(text);
    expect(segments).toHaveLength(1);
  });

  it('produces a single plain segment when there are no entities', () => {
    const text = 'just plain text';
    const segments = buildSegments(text, []);
    expect(reconstruct(segments)).toBe(text);
    expect(segments).toHaveLength(1);
    expect(segments[0].label).toBeNull();
  });

  it('handles an empty string', () => {
    const segments = buildSegments('', []);
    expect(segments).toEqual([]);
  });

  it('clips overlapping entities instead of throwing, and still reconstructs the text', () => {
    const text = 'John Smith Jones';
    const entities = [
      { label: 'PER', start: 0, end: 10 }, // "John Smith"
      { label: 'PER', start: 5, end: 16 }, // "Smith Jones" -- overlaps the first
    ];
    expect(() => buildSegments(text, entities)).not.toThrow();
    const segments = buildSegments(text, entities);
    expect(reconstruct(segments)).toBe(text);
  });

  it('drops entities with reversed or zero-length ranges', () => {
    const text = 'Hello world';
    const entities = [
      { label: 'PER', start: 5, end: 5 }, // zero-length
      { label: 'PER', start: 8, end: 3 }, // reversed
      { label: 'ORG', start: 0, end: 5 }, // valid: "Hello"
    ];
    const segments = buildSegments(text, entities);
    expect(reconstruct(segments)).toBe(text);
    const labeled = segments.filter((s) => s.label);
    expect(labeled).toHaveLength(1);
    expect(labeled[0]).toMatchObject({ label: 'ORG', text: 'Hello' });
  });

  it('clamps out-of-range start/end offsets instead of crashing', () => {
    const text = 'short';
    const entities = [{ label: 'PER', start: -5, end: 999 }];
    expect(() => buildSegments(text, entities)).not.toThrow();
    const segments = buildSegments(text, entities);
    expect(reconstruct(segments)).toBe(text);
  });

  it('ignores entities missing required fields', () => {
    const text = 'Hello world';
    const entities = [
      { label: 'PER' }, // missing start/end
      { start: 0, end: 5 }, // missing label
      null,
      undefined,
    ];
    expect(() => buildSegments(text, entities)).not.toThrow();
    const segments = buildSegments(text, entities);
    expect(reconstruct(segments)).toBe(text);
    expect(segments.every((s) => s.label === null)).toBe(true);
  });

  it('handles a non-array entities value gracefully', () => {
    const text = 'Hello world';
    expect(() => buildSegments(text, null)).not.toThrow();
    expect(reconstruct(buildSegments(text, null))).toBe(text);
    expect(reconstruct(buildSegments(text, undefined))).toBe(text);
  });
});

describe('<HighlightedText />', () => {
  it('renders the right number of highlighted spans with correct labels', () => {
    const text = 'John Smith works at Acme Corp.';
    const entities = [
      { label: 'PER', start: 0, end: 10 },
      { label: 'ORG', start: 20, end: 29 },
    ];
    render(<HighlightedText text={text} entities={entities} />);
    const spans = screen.getAllByTestId('entity-span');
    expect(spans).toHaveLength(2);
    expect(spans[0]).toHaveTextContent('John Smith');
    expect(spans[0]).toHaveClass('entity-PER');
    expect(spans[1]).toHaveTextContent('Acme Corp');
    expect(spans[1]).toHaveClass('entity-ORG');
  });

  it('renders plain text outside entities in the correct order', () => {
    const text = 'Paid by John Smith today.';
    const entities = [{ label: 'PER', start: 8, end: 18 }];
    render(<HighlightedText text={text} entities={entities} />);
    const container = screen.getByTestId('highlighted-text');
    expect(container).toHaveTextContent(text);
  });

  it('does not throw when given malformed/overlapping entity spans', () => {
    const text = 'Alpha Beta Gamma';
    const entities = [
      { label: 'PER', start: 0, end: 10 },
      { label: 'ORG', start: 6, end: 16 }, // overlaps
      { label: 'LOC', start: 100, end: 200 }, // fully out of range
      { label: 'DATE', start: 5, end: 2 }, // reversed
    ];
    expect(() =>
      render(<HighlightedText text={text} entities={entities} />),
    ).not.toThrow();
  });
});
