// API base URL is configurable via a Vite env var so the same build works
// both in local dev (backend on a different port, e.g. http://localhost:5000)
// and in production (Flask serves the frontend + API from the same origin,
// so the default empty string resolves to same-origin relative requests).
const RAW_BASE = import.meta.env.VITE_API_BASE_URL ?? '';
// Strip any trailing slash so `${API_BASE_URL}/predict` never produces `//predict`.
export const API_BASE_URL = RAW_BASE.replace(/\/+$/, '');

export class ApiError extends Error {
  constructor(message, { status, kind } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status ?? null;
    // 'network' | 'bad_input' | 'model_not_ready' | 'server' | 'parse'
    this.kind = kind ?? 'server';
  }
}

/**
 * Calls POST /predict with the given text and returns the parsed response
 * body on success. Throws ApiError with a human-readable message for every
 * failure mode: network failure, non-JSON body, 400, 503, or any other
 * non-2xx status.
 */
export async function predictEntities(text) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
  } catch (err) {
    throw new ApiError(
      'Could not reach the analysis server. Check your connection and try again.',
      { kind: 'network' },
    );
  }

  let body = null;
  let parseFailed = false;
  try {
    body = await response.json();
  } catch (err) {
    parseFailed = true;
  }

  if (!response.ok) {
    const serverMessage =
      body && typeof body.error === 'string' ? body.error : null;

    if (response.status === 400) {
      throw new ApiError(
        serverMessage ||
          'The submitted text could not be processed. Please check your input.',
        { status: 400, kind: 'bad_input' },
      );
    }

    if (response.status === 503) {
      throw new ApiError(
        serverMessage ||
          'The model is not ready yet. Please try again in a moment.',
        { status: 503, kind: 'model_not_ready' },
      );
    }

    throw new ApiError(
      serverMessage || `Request failed with status ${response.status}.`,
      { status: response.status, kind: 'server' },
    );
  }

  if (parseFailed) {
    throw new ApiError(
      'The server returned a response that could not be understood.',
      { status: response.status, kind: 'parse' },
    );
  }

  return body;
}
