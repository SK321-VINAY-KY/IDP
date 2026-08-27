import "./SchemaBuilder.css";

export default function SchemaBuilder({ fields, onChange, disabled }) {
  function updateField(index, key, value) {
    const next = fields.slice();
    next[index] = { ...next[index], [key]: value };
    onChange(next);
  }

  function addField() {
    onChange([...fields, { name: "", description: "" }]);
  }

  function removeField(index) {
    const next = fields.filter((_, i) => i !== index);
    onChange(next.length ? next : [{ name: "", description: "" }]);
  }

  return (
    <div className="schema-builder">
      <div className="schema-head">
        <span>Field name</span>
        <span>What to look for</span>
        <span aria-hidden="true" />
      </div>
      {fields.map((field, i) => (
        <div className="schema-row" key={i}>
          <input
            className="schema-input schema-name"
            placeholder="invoice_number"
            value={field.name}
            disabled={disabled}
            onChange={(e) => updateField(i, "name", e.target.value)}
          />
          <input
            className="schema-input"
            placeholder="The unique invoice ID printed near the header"
            value={field.description}
            disabled={disabled}
            onChange={(e) => updateField(i, "description", e.target.value)}
          />
          <button
            type="button"
            className="schema-remove"
            onClick={() => removeField(i)}
            disabled={disabled}
            aria-label="Remove field"
          >
            ×
          </button>
        </div>
      ))}
      <button type="button" className="schema-add" onClick={addField} disabled={disabled}>
        + Add field
      </button>
    </div>
  );
}
