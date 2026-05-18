import { useState, useEffect } from "react";
import { getHistory } from "../services/api";

const CLASS_COLORS = {
  dent:          "#ff6347",
  scratch:       "#1e90ff",
  crack:         "#32cd32",
  glass_shatter: "#9400d3",
  lamp_broken:   "#ffd700",
  tire_flat:     "#ff4500",
};

const CLASS_EMOJIS = {
  dent:          "🔴",
  scratch:       "🔵",
  crack:         "🟢",
  glass_shatter: "🟣",
  lamp_broken:   "🟡",
  tire_flat:     "🟠",
};

const SEVERITY_COLORS = {
  Minor:    "#22c55e",
  Moderate: "#f59e0b",
  Severe:   "#ef4444",
  Rejected: "#64748b",
  None:     "#64748b",
};

export default function History() {
  const [detections, setDetections] = useState([]);
  const [loading,    setLoading]    = useState(true);
  const [error,      setError]      = useState(null);
  const [skip,       setSkip]       = useState(0);
  const LIMIT = 20;

  useEffect(() => {
    fetchHistory();
  }, [skip]);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getHistory(skip, LIMIT);
      setDetections(Array.isArray(data) ? data : []);
    } catch (err) {
      setError("Failed to load history. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateStr) => new Date(dateStr).toLocaleString();

  // Get unique damage classes from results array
  const getDamageTypes = (results) => {
    if (!results || !Array.isArray(results)) return [];
    return [...new Set(results.map((r) => r.class_name))];
  };

  // Get overall severity from results array
  const getOverallSeverity = (results) => {
    if (!results || results.length === 0) return null;
    const scores = results.map((r) => r.severity_score || 0);
    const avg    = scores.reduce((a, b) => a + b, 0) / scores.length;
    if (avg < 0.35) return "Minor";
    if (avg < 0.65) return "Moderate";
    return "Severe";
  };

  return (
    <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "2rem" }}>

      {/* Header */}
      <div style={{ marginBottom: "2rem" }}>
        <h1 style={{ fontSize: "1.8rem", fontWeight: "700", color: "#f1f5f9" }}>
          🕓 Detection History
        </h1>
        <p style={{ color: "#64748b", marginTop: "0.5rem" }}>
          All past car damage detection results
        </p>
      </div>

      {/* Loading */}
      {loading && (
        <div style={{ textAlign: "center", color: "#64748b", padding: "3rem" }}>
          <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>⏳</div>
          <p>Loading history...</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          background:   "#450a0a",
          border:       "1px solid #ef4444",
          borderRadius: "8px",
          padding:      "1rem",
          color:        "#fca5a5",
          fontSize:     "0.9rem",
        }}>
          ❌ {error}
        </div>
      )}

      {/* Empty */}
      {!loading && !error && detections.length === 0 && (
        <div style={{ textAlign: "center", color: "#64748b", padding: "3rem" }}>
          <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🚗</div>
          <p>No detections yet. Upload some images to get started!</p>
        </div>
      )}

      {/* Table */}
      {!loading && detections.length > 0 && (
        <div style={{
          background:   "#1e293b",
          borderRadius: "12px",
          border:       "1px solid #334155",
          overflow:     "hidden",
        }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "#0f172a" }}>
                {["ID", "Image", "Instances", "Damage Types", "Severity", "Model", "Date"].map((h) => (
                  <th key={h} style={{
                    padding:       "0.85rem 1rem",
                    textAlign:     "left",
                    fontSize:      "0.8rem",
                    color:         "#64748b",
                    fontWeight:    "600",
                    letterSpacing: "0.05em",
                    textTransform: "uppercase",
                    borderBottom:  "1px solid #334155",
                  }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {detections.map((det, index) => {
                const damageTypes     = getDamageTypes(det.results);
                const overallSeverity = getOverallSeverity(det.results);
                const severityColor   = SEVERITY_COLORS[overallSeverity] || "#64748b";

                return (
                  <tr
                    key={det.id}
                    style={{
                      borderBottom: "1px solid #1e293b",
                      background:   index % 2 === 0 ? "#1e293b" : "#172033",
                    }}
                  >
                    {/* ID */}
                    <td style={{ padding: "0.85rem 1rem", color: "#64748b", fontSize: "0.85rem" }}>
                      #{det.id}
                    </td>

                    {/* Image */}
                    <td style={{
                      padding:      "0.85rem 1rem",
                      color:        "#e2e8f0",
                      fontSize:     "0.85rem",
                      maxWidth:     "180px",
                      overflow:     "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace:   "nowrap",
                    }}>
                      📄 {det.image_filename}
                    </td>

                    {/* Instances */}
                    <td style={{ padding: "0.85rem 1rem" }}>
                      <span style={{
                        background:   "#1d4ed8",
                        color:        "#bfdbfe",
                        borderRadius: "12px",
                        padding:      "2px 10px",
                        fontSize:     "0.8rem",
                        fontWeight:   "600",
                      }}>
                        {det.total_instances}
                      </span>
                    </td>

                    {/* Damage Types */}
                    <td style={{ padding: "0.85rem 1rem" }}>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                        {damageTypes.length > 0
                          ? damageTypes.map((cls) => (
                              <span key={cls} style={{
                                background:   (CLASS_COLORS[cls] || "#64748b") + "22",
                                color:        CLASS_COLORS[cls] || "#64748b",
                                border:       `1px solid ${(CLASS_COLORS[cls] || "#64748b")}44`,
                                borderRadius: "12px",
                                padding:      "2px 8px",
                                fontSize:     "0.75rem",
                                fontWeight:   "500",
                              }}>
                                {CLASS_EMOJIS[cls]} {cls.replace(/_/g, " ")}
                              </span>
                            ))
                          : <span style={{ color: "#64748b", fontSize: "0.8rem" }}>—</span>
                        }
                      </div>
                    </td>

                    {/* Severity */}
                    <td style={{ padding: "0.85rem 1rem" }}>
                      {overallSeverity ? (
                        <span style={{
                          background:   severityColor + "22",
                          color:        severityColor,
                          border:       `1px solid ${severityColor}66`,
                          borderRadius: "12px",
                          padding:      "2px 10px",
                          fontSize:     "0.78rem",
                          fontWeight:   "700",
                        }}>
                          {overallSeverity}
                        </span>
                      ) : (
                        <span style={{ color: "#64748b", fontSize: "0.8rem" }}>—</span>
                      )}
                    </td>

                    {/* Model */}
                    <td style={{ padding: "0.85rem 1rem", color: "#64748b", fontSize: "0.8rem" }}>
                      {det.model_version}
                    </td>

                    {/* Date */}
                    <td style={{ padding: "0.85rem 1rem", color: "#64748b", fontSize: "0.8rem" }}>
                      {formatDate(det.created_at)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {/* Pagination */}
          <div style={{
            display:        "flex",
            justifyContent: "space-between",
            alignItems:     "center",
            padding:        "1rem 1.5rem",
            borderTop:      "1px solid #334155",
          }}>
            <span style={{ color: "#64748b", fontSize: "0.85rem" }}>
              Showing {skip + 1}–{skip + detections.length} results
            </span>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button
                onClick={() => setSkip(Math.max(0, skip - LIMIT))}
                disabled={skip === 0}
                style={{
                  background:   skip === 0 ? "#0f172a" : "#1e293b",
                  color:        skip === 0 ? "#334155" : "#94a3b8",
                  border:       "1px solid #334155",
                  borderRadius: "6px",
                  padding:      "0.4rem 1rem",
                  cursor:       skip === 0 ? "not-allowed" : "pointer",
                  fontSize:     "0.85rem",
                }}
              >
                ← Previous
              </button>
              <button
                onClick={() => setSkip(skip + LIMIT)}
                disabled={detections.length < LIMIT}
                style={{
                  background:   detections.length < LIMIT ? "#0f172a" : "#1e293b",
                  color:        detections.length < LIMIT ? "#334155" : "#94a3b8",
                  border:       "1px solid #334155",
                  borderRadius: "6px",
                  padding:      "0.4rem 1rem",
                  cursor:       detections.length < LIMIT ? "not-allowed" : "pointer",
                  fontSize:     "0.85rem",
                }}
              >
                Next →
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}