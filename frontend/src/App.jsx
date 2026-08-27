import { useState } from "react";
import ExtractWorkflow from "./components/ExtractWorkflow.jsx";
import "./App.css";

export default function App() {
  const [view, setView] = useState("user");

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">IDP</span>
          <span className="brand-sub">Intelligent Document Processing</span>
        </div>
        <nav className="topnav">
          <button
            className={view === "user" ? "topnav-tab active" : "topnav-tab"}
            onClick={() => setView("user")}
          >
            Extract
          </button>
          <button
            className="topnav-tab disabled"
            onClick={() => setView("admin")}
            aria-disabled="true"
          >
            Admin <span className="soon">soon</span>
          </button>
        </nav>
      </header>

      <main className="stage">
        {view === "user" ? (
          <ExtractWorkflow />
        ) : (
          <div className="admin-stub">
            <p className="admin-stub-eyebrow">Admin console</p>
            <h2>Not open yet.</h2>
            <p>Schema libraries, run history, and access control land here next.</p>
          </div>
        )}
      </main>
    </div>
  );
}
