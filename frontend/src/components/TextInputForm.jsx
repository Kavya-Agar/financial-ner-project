import { useId } from 'react';

export default function TextInputForm({ text, onTextChange, onSubmit, loading }) {
  const textAreaId = useId();

  function handleFileChange(event) {
    const file = event.target.files && event.target.files[0];
    // Always clear the input value so selecting the same file twice in a
    // row still fires a change event.
    event.target.value = '';
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      onTextChange(String(e.target?.result ?? ''));
    };
    reader.readAsText(file);
  }

  function handleSubmit(event) {
    event.preventDefault();
    onSubmit(text);
  }

  const canSubmit = !loading && text.trim().length > 0;

  return (
    <form className="input-form" onSubmit={handleSubmit}>
      <label htmlFor={textAreaId} className="field-label">
        Paste text
      </label>
      <textarea
        id={textAreaId}
        value={text}
        onChange={(e) => onTextChange(e.target.value)}
        placeholder="Paste bank statement or tax form text here..."
        rows={10}
        disabled={loading}
      />
      <div className="form-controls">
        <label className="file-label">
          Upload .txt file
          <input
            type="file"
            accept=".txt,text/plain"
            onChange={handleFileChange}
            disabled={loading}
            aria-label="Upload a .txt file"
          />
        </label>
        <button type="submit" disabled={!canSubmit}>
          {loading ? 'Analyzing…' : 'Analyze'}
        </button>
      </div>
    </form>
  );
}
