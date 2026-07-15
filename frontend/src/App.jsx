import { useCallback, useState } from 'react';
import TextInputForm from './components/TextInputForm.jsx';
import HighlightedText from './components/HighlightedText.jsx';
import Legend from './components/Legend.jsx';
import { predictEntities, ApiError } from './api.js';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleSubmit = useCallback(async (submittedText) => {
    const trimmed = (submittedText ?? '').trim();
    if (!trimmed) {
      setError('Please enter some text or upload a .txt file before analyzing.');
      setResult(null);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await predictEntities(submittedText);
      setResult(data);
    } catch (err) {
      setResult(null);
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Something went wrong while analyzing the text. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Financial Document NER</h1>
        <p className="subtitle">
          Paste financial document text (bank statements, tax forms) or
          upload a .txt file to extract named entities.
        </p>
      </header>

      <TextInputForm
        text={text}
        onTextChange={setText}
        onSubmit={handleSubmit}
        loading={loading}
      />

      {loading && (
        <div className="status status-loading" role="status">
          <span className="spinner" aria-hidden="true" />
          Analyzing…
        </div>
      )}

      {error && !loading && (
        <div className="status status-error" role="alert">
          {error}
        </div>
      )}

      {!loading && !error && result && (
        <section className="results">
          <Legend />
          {Array.isArray(result.entities) && result.entities.length > 0 ? (
            <HighlightedText text={result.text} entities={result.entities} />
          ) : (
            <p className="no-entities">No entities found.</p>
          )}
          {result.truncated && (
            <p className="truncated-note">
              Note: the input text was truncated before analysis.
            </p>
          )}
        </section>
      )}
    </div>
  );
}

export default App;
