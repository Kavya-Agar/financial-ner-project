import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App.jsx';

function jsonResponse(body, { ok = true, status = 200 } = {}) {
  return {
    ok,
    status,
    json: () => Promise.resolve(body),
  };
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

async function typeAndSubmit(user, text) {
  const textarea = screen.getByLabelText(/paste text/i);
  await user.clear(textarea);
  await user.type(textarea, text);
  await user.click(screen.getByRole('button', { name: /analyze/i }));
}

describe('<App /> success flow', () => {
  it('renders the right number of highlighted spans, labels, and legend entries for a successful response', async () => {
    const user = userEvent.setup();
    const text = 'John Smith paid $500 on 2024-01-05.';
    fetch.mockResolvedValueOnce(
      jsonResponse({
        text,
        truncated: false,
        entities: [
          { text: 'John Smith', label: 'PER', start: 0, end: 10, score: 0.98 },
          { text: '$500', label: 'AMOUNT', start: 16, end: 20, score: 0.9 },
          { text: '2024-01-05', label: 'DATE', start: 24, end: 34, score: 0.99 },
        ],
      }),
    );

    render(<App />);
    await typeAndSubmit(user, text);

    await waitFor(() => {
      expect(screen.getAllByTestId('entity-span')).toHaveLength(3);
    });
    const spans = screen.getAllByTestId('entity-span');
    expect(spans.map((s) => s.dataset.label)).toEqual(['PER', 'AMOUNT', 'DATE']);

    // Legend always shows all 8 entity types.
    expect(screen.getAllByTestId('legend-item')).toHaveLength(8);
    expect(screen.getByText('Person')).toBeInTheDocument();
    expect(screen.getByText('Amount')).toBeInTheDocument();
    expect(screen.getByText('SSN')).toBeInTheDocument();

    // Full text (highlighted + plain) reconstructs exactly.
    expect(screen.getByTestId('highlighted-text')).toHaveTextContent(text);
  });

  it('shows "No entities found" when entities is an empty array', async () => {
    const user = userEvent.setup();
    const text = 'Nothing interesting here.';
    fetch.mockResolvedValueOnce(
      jsonResponse({ text, truncated: false, entities: [] }),
    );

    render(<App />);
    await typeAndSubmit(user, text);

    await waitFor(() => {
      expect(screen.getByText(/no entities found/i)).toBeInTheDocument();
    });
    expect(screen.queryByTestId('entity-span')).not.toBeInTheDocument();
  });
});

describe('<App /> loading state', () => {
  it('disables the submit button and shows a loading indicator while the request is pending', async () => {
    const user = userEvent.setup();
    let resolveFetch;
    fetch.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveFetch = resolve;
      }),
    );

    render(<App />);
    const textarea = screen.getByLabelText(/paste text/i);
    await user.type(textarea, 'Some text to analyze');
    const button = screen.getByRole('button', { name: /analyze/i });
    await user.click(button);

    expect(screen.getByRole('button', { name: /analyzing/i })).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent(/analyzing/i);

    resolveFetch(
      jsonResponse({ text: 'Some text to analyze', truncated: false, entities: [] }),
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /^analyze$/i })).not.toBeDisabled();
    });
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });
});

describe('<App /> error states', () => {
  it('shows an error message on a network failure and does not crash', async () => {
    const user = userEvent.setup();
    fetch.mockRejectedValueOnce(new TypeError('Failed to fetch'));

    render(<App />);
    await typeAndSubmit(user, 'Some text');

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/could not reach/i);
    });
  });

  it('shows an error message when the response body is not valid JSON', async () => {
    const user = userEvent.setup();
    fetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.reject(new SyntaxError('Unexpected token')),
    });

    render(<App />);
    await typeAndSubmit(user, 'Some text');

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });
    expect(screen.getByRole('alert')).toHaveTextContent(/could not be understood/i);
  });

  it('shows a "bad input" message for a 400 response', async () => {
    const user = userEvent.setup();
    fetch.mockResolvedValueOnce(
      jsonResponse(
        { error: 'text field is required' },
        { ok: false, status: 400 },
      ),
    );

    render(<App />);
    await typeAndSubmit(user, 'Some text');

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('text field is required');
    });
  });

  it('shows a "model not ready" message for a 503 response', async () => {
    const user = userEvent.setup();
    fetch.mockResolvedValueOnce(
      jsonResponse(
        { error: 'model is still loading' },
        { ok: false, status: 503 },
      ),
    );

    render(<App />);
    await typeAndSubmit(user, 'Some text');

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('model is still loading');
    });
  });

  it('falls back to a generic message for a 500 response with no error body', async () => {
    const user = userEvent.setup();
    fetch.mockResolvedValueOnce(jsonResponse({}, { ok: false, status: 500 }));

    render(<App />);
    await typeAndSubmit(user, 'Some text');

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/status 500/i);
    });
  });

  it('does not render stale results alongside a new error', async () => {
    const user = userEvent.setup();
    fetch.mockResolvedValueOnce(
      jsonResponse({
        text: 'Acme Corp',
        truncated: false,
        entities: [{ label: 'ORG', start: 0, end: 9, text: 'Acme Corp' }],
      }),
    );
    render(<App />);
    await typeAndSubmit(user, 'Acme Corp');
    await waitFor(() => {
      expect(screen.getAllByTestId('entity-span')).toHaveLength(1);
    });

    fetch.mockResolvedValueOnce(
      jsonResponse({ error: 'bad' }, { ok: false, status: 400 }),
    );
    await typeAndSubmit(user, 'more text');

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });
    expect(screen.queryByTestId('entity-span')).not.toBeInTheDocument();
  });
});

describe('<App /> file upload', () => {
  it('populates the textarea from an uploaded .txt file', async () => {
    const user = userEvent.setup();
    render(<App />);

    const fileContents = 'Text loaded from a file.';
    const file = new File([fileContents], 'sample.txt', { type: 'text/plain' });
    const input = screen.getByLabelText(/upload a \.txt file/i);

    await user.upload(input, file);

    await waitFor(() => {
      expect(screen.getByLabelText(/paste text/i)).toHaveValue(fileContents);
    });
  });
});
