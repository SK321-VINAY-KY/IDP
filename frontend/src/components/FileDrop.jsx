import { useCallback, useRef, useState } from "react";
import "./FileDrop.css";

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function FileDrop({ file, onSelect, disabled }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [rejection, setRejection] = useState("");

  const acceptFile = useCallback(
    (candidate) => {
      if (!candidate) return;
      if (candidate.type !== "application/pdf" && !candidate.name.toLowerCase().endsWith(".pdf")) {
        setRejection("Only PDF files are accepted.");
        return;
      }
      setRejection("");
      onSelect(candidate);
    },
    [onSelect]
  );

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    acceptFile(e.dataTransfer.files?.[0]);
  }

  return (
    <div>
      <div
        className={`dropzone${dragging ? " dragging" : ""}${file ? " filled" : ""}${
          disabled ? " disabled" : ""
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !disabled) inputRef.current?.click();
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          hidden
          disabled={disabled}
          onChange={(e) => acceptFile(e.target.files?.[0])}
        />
        {file ? (
          <div className="dropzone-file">
            <span className="dropzone-filename">{file.name}</span>
            <span className="dropzone-filesize">{formatSize(file.size)}</span>
          </div>
        ) : (
          <>
            <p className="dropzone-title">Drop a PDF here, or click to browse</p>
            <p className="dropzone-sub">One document per run</p>
          </>
        )}
      </div>
      {rejection && <p className="dropzone-error">{rejection}</p>}
    </div>
  );
}
