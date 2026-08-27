import { useState } from "react";
import "./ResultTicket.css";

function download(filename, content) {
  const blob = new Blob([content], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ResultTicket({ result, fileName }) {
  const [showRaw, setShowRaw] = useState(false);
  const data = result?.data || {};
  const meta = result?.meta || {};
  const lowConfidencePages = meta.low_confidence_pages || [];

  return (
    <section className="ticket">
      <div className="ticket-stamp">processed</div>
      <header className="ticket-header">
        <p className="ticket-eyebrow">04 — Extraction manifest</p>
        <h3 className="ticket-source">{fileName}</h3>
      </header>

      <dl className="ticket-fields">
        {Object.keys(data).length === 0 && (
          <p className="ticket-empty">No fields came back for this schema.</p>
        )}
        {Object.entries(data).map(([key, value]) => (
          <div className="ticket-field" key={key}>
            <dt>{key}</dt>
            <dd>{value === "" || value === null ? <em>empty</em> : String(value)}</dd>
          </div>
        ))}
      </dl>

      <div className="ticket-tear" aria-hidden="true" />

      <div className="ticket-meta">
        <span className="meta-chip">{meta.page_count ?? "?"} page{meta.page_count === 1 ? "" : "s"}</span>
        <span className="meta-chip">{meta.strategy_used || "—"}</span>
        <span className="meta-chip">{meta.processing_time_seconds ?? "?"}s</span>
        {(meta.engines_used || []).map((engine) => (
          <span className="meta-chip" key={engine}>
            {engine}
          </span>
        ))}
        {lowConfidencePages.length > 0 && (
          <span className="meta-chip warn">
            low confidence — page{lowConfidencePages.length > 1 ? "s" : ""}{" "}
            {lowConfidencePages.join(", ")}
          </span>
        )}
      </div>

      <div className="ticket-actions">
        <button className="ticket-action" onClick={() => setShowRaw((v) => !v)}>
          {showRaw ? "Hide raw JSON" : "View raw JSON"}
        </button>
        <button
          className="ticket-action"
          onClick={() => download(`${(fileName || "extraction").replace(/\.pdf$/i, "")}.json`, JSON.stringify(result, null, 2))}
        >
          Download JSON
        </button>
      </div>

      {showRaw && <pre className="ticket-raw">{JSON.stringify(result, null, 2)}</pre>}
    </section>
  );
}
