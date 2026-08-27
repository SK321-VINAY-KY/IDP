import { useMemo, useState } from "react";
import FileDrop from "./FileDrop.jsx";
import SchemaBuilder from "./SchemaBuilder.jsx";
import ResultTicket from "./ResultTicket.jsx";
import { runExtraction, ApiError } from "../lib/api.js";
import "./ExtractWorkflow.css";

const STEPS = [
  { id: "upload", label: "Upload document" },
  { id: "schema", label: "Define target schema" },
  { id: "run", label: "Run extraction" },
];

export default function ExtractWorkflow() {
  const [file, setFile] = useState(null);
  const [fields, setFields] = useState([{ name: "", description: "" }]);
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const validFields = useMemo(
    () => fields.filter((f) => f.name.trim().length > 0),
    [fields]
  );

  const canRun = Boolean(file) && validFields.length > 0 && status !== "running";

  const activeStep = !file ? 0 : validFields.length === 0 ? 1 : status === "idle" ? 2 : 2;

  async function handleRun() {
    setStatus("running");
    setError("");
    setResult(null);
    try {
      const payload = await runExtraction(file, validFields);
      setResult(payload);
      setStatus("done");
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Something went wrong while extracting this document."
      );
      setStatus("error");
    }
  }

  function handleReset() {
    setFile(null);
    setFields([{ name: "", description: "" }]);
    setStatus("idle");
    setResult(null);
    setError("");
  }

  return (
    <div className="workflow">
      <ol className="manifest-rail" aria-hidden="true">
        {STEPS.map((step, i) => (
          <li key={step.id} className={i <= activeStep ? "rail-step done" : "rail-step"}>
            <span className="rail-num">{String(i + 1).padStart(2, "0")}</span>
            <span className="rail-label">{step.label}</span>
          </li>
        ))}
      </ol>

      <div className="workflow-main">
        <section className="panel">
          <p className="panel-eyebrow">01 — Source document</p>
          <FileDrop file={file} onSelect={setFile} disabled={status === "running"} />
        </section>

        <section className="panel">
          <p className="panel-eyebrow">02 — Target schema</p>
          <p className="panel-hint">
            List the fields you want pulled out. Each one becomes a key in the response.
          </p>
          <SchemaBuilder fields={fields} onChange={setFields} disabled={status === "running"} />
        </section>

        <section className="panel run-panel">
          <p className="panel-eyebrow">03 — Run</p>
          <div className="run-row">
            <button
              className="run-button"
              onClick={handleRun}
              disabled={!canRun}
            >
              {status === "running" ? "Extracting…" : "Extract document"}
            </button>
            {(file || validFields.length > 0 || status !== "idle") && (
              <button className="reset-button" onClick={handleReset} disabled={status === "running"}>
                Start over
              </button>
            )}
          </div>
          {status === "running" && (
            <p className="run-status">
              Routing pages, running OCR where needed, and matching your schema against the
              content. This can take a while for longer documents.
            </p>
          )}
          {status === "error" && <p className="run-error">{error}</p>}
        </section>

        {status === "done" && result && <ResultTicket result={result} fileName={file?.name} />}
      </div>
    </div>
  );
}
