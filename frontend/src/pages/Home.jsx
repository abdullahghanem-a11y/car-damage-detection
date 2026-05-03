import { useState } from "react";
import UploadZone from "../components/UploadZone";
import ResultCard from "../components/ResultCard";
import { detectDamage } from "../services/api";

export default function Home() {
  const [files,   setFiles]   = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const handleDetect = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const data = await detectDamage(files);
      setResults(data);
    } catch (err) {
      setError(
        err.response?.data?.detail || "Something went wrong. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFiles([]);
    setResults(null);
    setError(null);
  };

  return (
    <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "2rem" }}>

      {/* Header */}
      <div style={{ marginBottom: "2rem" }}>
        <h1 style={{ fontSize: "1.8rem", fontWeight: "700", color: "#f1f5f9" }}>
          Detect Car Damage
        </h1>
        <p style={{ color: "#64748b", marginTop: "0.5rem" }}>
          Upload up to 10 car images to automatically detect and classify damage
        </p>
      </div>

      {/* Upload Zone */}
      {!results && (
        <UploadZone onFilesSelected={setFiles} />
      )}

      {/* Error */}
      {error && (
        <div style={{
          background:   "#450a0a",
          border:       "1px solid #ef4444",
          borderRadius: "8px",
          padding:      "1rem",
          marginTop:    "1rem",
          color:        "#fca5a5",
          fontSize:     "0.9rem",
        }}>
          Error: {error}
        </div>
      )}

      {/* Detect Button */}
      {!results && (
        <div style={{ marginTop: "1.5rem", display: "flex", gap: "1rem" }}>
          <button
            onClick={handleDetect}
            disabled={files.length === 0 || loading}
            style={{
              background:   files.length === 0 || loading ? "#1e293b" : "#3b82f6",
              color:        files.length === 0 || loading ? "#475569" : "white",
              border:       "none",
              borderRadius: "8px",
              padding:      "0.75rem 2rem",
              fontSize:     "0.95rem",
              fontWeight:   "600",
              cursor:       files.length === 0 || loading ? "not-allowed" : "pointer",
              transition:   "background 0.2s",
            }}
          >
            {loading
              ? "Detecting..."
              : files.length > 0
                ? "Detect Damage (" + files.length + " image" + (files.length > 1 ? "s" : "") + ")"
                : "Detect Damage"}
          </button>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div style={{ marginTop: "2rem", textAlign: "center", color: "#64748b" }}>
          <p>Running YOLOv8 inference on {files.length} image{files.length > 1 ? "s" : ""}...</p>
        </div>
      )}

      {/* Results */}
      {results && (
        <div>
          {/* Summary */}
          <div style={{
            display:        "flex",
            alignItems:     "center",
            justifyContent: "space-between",
            marginBottom:   "1.5rem",
            flexWrap:       "wrap",
            gap:            "1rem",
          }}>
            <div>
              <h2 style={{ fontSize: "1.3rem", fontWeight: "700", color: "#f1f5f9" }}>
                Detection Complete
              </h2>
              <p style={{ color: "#64748b", marginTop: "0.25rem", fontSize: "0.9rem" }}>
                {results.total_images} image{results.total_images > 1 ? "s" : ""} processed
                {" — "}
                {results.total_instances} damage instance{results.total_instances !== 1 ? "s" : ""} found
              </p>
            </div>
            <button
              onClick={handleReset}
              style={{
                background:   "#1e293b",
                color:        "#94a3b8",
                border:       "1px solid #334155",
                borderRadius: "8px",
                padding:      "0.6rem 1.25rem",
                cursor:       "pointer",
                fontSize:     "0.9rem",
              }}
            >
              Detect New Images
            </button>
          </div>

          {/* Result Cards Grid */}
          <div style={{
            display:             "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))",
            gap:                 "1.5rem",
          }}>
            {results.results.map((result, index) => (
              <ResultCard key={index} result={result} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}