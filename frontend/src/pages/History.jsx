import { useState, useEffect } from "react";
import { getHistory } from "../services/api";

const CLASS_COLORS = {
  dent: "#ff6347", scratch: "#3b82f6", crack: "#22c55e",
  glass_shatter: "#a855f7", lamp_broken: "#eab308", tire_flat: "#f97316",
};
const CLASS_LABELS = {
  dent: "Dent", scratch: "Scratch", crack: "Crack",
  glass_shatter: "Glass Shatter", lamp_broken: "Lamp Broken", tire_flat: "Tire Flat",
};
const SEVERITY_COLORS = {
  Minor: "#22c55e", Moderate: "#f59e0b", Severe: "#ef4444",
  Rejected: "#64748b", None: "#64748b",
};

export default function History() {
  const [detections, setDetections] = useState([]);
  const [loading,    setLoading]    = useState(true);
  const [error,      setError]      = useState(null);
  const [skip,       setSkip]       = useState(0);
  const LIMIT = 20;

  useEffect(() => { fetchHistory(); }, [skip]);

  const fetchHistory = async () => {
    setLoading(true); setError(null);
    try {
      const data = await getHistory(skip, LIMIT);
      setDetections(Array.isArray(data) ? data : []);
    } catch {
      setError("Failed to load history. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const formatDate     = (d) => new Date(d).toLocaleString();
  const getDamageTypes = (r) => r ? [...new Set(r.map((x) => x.class_name))] : [];
  const getOverallSev  = (r) => {
    if (!r?.length) return null;
    const avg = r.reduce((a, b) => a + (b.severity_score || 0), 0) / r.length;
    return avg < 0.35 ? "Minor" : avg < 0.65 ? "Moderate" : "Severe";
  };

  return (
    <>
      <style>{`
        /* Mobile card layout */
        .history-card {
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 0.85rem;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          transition: border-color 0.15s;
        }
        .history-card:hover { border-color: var(--text-faint); }
        .history-row-label {
          font-size: 0.65rem;
          color: var(--text-faint);
          letter-spacing: 0.05em;
          text-transform: uppercase;
          margin-bottom: 2px;
        }

        /* Desktop table */
        .history-table { width: 100%; border-collapse: collapse; }
        .history-table tr { transition: background 0.15s; }
        .history-table tr:hover td { background: var(--bg-hover) !important; }

        .page-btn {
          background: transparent; color: var(--text-muted);
          border: 1px solid var(--border); border-radius: 6px;
          padding: 0.4rem 0.85rem; cursor: pointer; font-size: 0.8rem;
          transition: all 0.2s;
        }
        .page-btn:hover:not(:disabled) { border-color: var(--text-faint); color: var(--text-secondary); }
        .page-btn:disabled { opacity: 0.3; cursor: not-allowed; }

        /* Show table on desktop, cards on mobile */
        .table-wrapper { display: block; }
        .cards-wrapper  { display: none; }

        @media (max-width: 600px) {
          .table-wrapper { display: none; }
          .cards-wrapper  { display: flex; flex-direction: column; gap: 0.65rem; }
        }
      `}</style>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "1rem" }}>

        <div style={{ marginBottom: "1.5rem" }}>
          <h1 style={{ fontSize: "1.4rem", fontWeight: "700", color: "var(--text-primary)" }}>
            Detection History
          </h1>
          <p style={{ color: "var(--text-muted)", marginTop: "0.3rem", fontSize: "0.85rem" }}>
            All past car damage detection results
          </p>
        </div>

        {loading && (
          <div style={{ textAlign: "center", color: "var(--text-muted)", padding: "2.5rem" }}>
            <p style={{ fontSize: "0.88rem" }}>Loading history...</p>
          </div>
        )}

        {error && (
          <div style={{
            background: "var(--bg-elevated)", border: "1px solid var(--danger)",
            borderRadius: "8px", padding: "0.75rem",
            color: "var(--danger)", fontSize: "0.82rem",
          }}>
            {error}
          </div>
        )}

        {!loading && !error && detections.length === 0 && (
          <div style={{ textAlign: "center", color: "var(--text-faint)", padding: "2.5rem" }}>
            <p style={{ fontSize: "2rem", marginBottom: "0.6rem" }}>🚗</p>
            <p style={{ fontSize: "0.88rem" }}>No detections yet.</p>
          </div>
        )}

        {!loading && detections.length > 0 && (
          <>
            {/* ── Desktop Table ── */}
            <div className="table-wrapper" style={{
              background: "var(--bg-surface)", borderRadius: "12px",
              border: "1px solid var(--border)", overflow: "hidden",
              boxShadow: "var(--shadow)",
            }}>
              <table className="history-table">
                <thead>
                  <tr style={{ background: "var(--bg-elevated)" }}>
                    {["ID", "Image", "Instances", "Damage Types", "Severity", "Model", "Date"].map((h) => (
                      <th key={h} style={{
                        padding: "0.75rem 0.85rem", textAlign: "left",
                        fontSize: "0.68rem", color: "var(--text-faint)",
                        fontWeight: "700", letterSpacing: "0.06em",
                        textTransform: "uppercase", borderBottom: "1px solid var(--border)",
                      }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {detections.map((det) => {
                    const dmgTypes = getDamageTypes(det.results);
                    const sev      = getOverallSev(det.results);
                    const sevColor = SEVERITY_COLORS[sev] || "#64748b";
                    return (
                      <tr key={det.id}>
                        <td style={{ padding: "0.75rem 0.85rem", color: "var(--text-faint)", fontSize: "0.78rem", background: "var(--bg-surface)" }}>#{det.id}</td>
                        <td style={{ padding: "0.75rem 0.85rem", color: "var(--text-secondary)", fontSize: "0.8rem", maxWidth: "140px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", background: "var(--bg-surface)" }}>
                          {det.image_filename}
                        </td>
                        <td style={{ padding: "0.75rem 0.85rem", background: "var(--bg-surface)" }}>
                          <span style={{
                            background: "var(--accent)22", color: "var(--accent)",
                            border: "1px solid var(--accent)44", borderRadius: "6px",
                            padding: "1px 7px", fontSize: "0.73rem", fontWeight: "600",
                          }}>
                            {det.total_instances}
                          </span>
                        </td>
                        <td style={{ padding: "0.75rem 0.85rem", background: "var(--bg-surface)" }}>
                          <div style={{ display: "flex", flexWrap: "wrap", gap: "3px" }}>
                            {dmgTypes.length > 0
                              ? dmgTypes.map((cls) => (
                                  <span key={cls} style={{
                                    background: (CLASS_COLORS[cls] || "#64748b") + "18",
                                    color: CLASS_COLORS[cls] || "#64748b",
                                    border: `1px solid ${(CLASS_COLORS[cls] || "#64748b")}33`,
                                    borderRadius: "6px", padding: "1px 6px",
                                    fontSize: "0.7rem", fontWeight: "500",
                                  }}>
                                    {CLASS_LABELS[cls] || cls.replace(/_/g, " ")}
                                  </span>
                                ))
                              : <span style={{ color: "var(--text-faint)", fontSize: "0.78rem" }}>—</span>
                            }
                          </div>
                        </td>
                        <td style={{ padding: "0.75rem 0.85rem", background: "var(--bg-surface)" }}>
                          {sev ? (
                            <span style={{
                              background: sevColor + "18", color: sevColor,
                              border: `1px solid ${sevColor}44`, borderRadius: "6px",
                              padding: "1px 7px", fontSize: "0.7rem", fontWeight: "700",
                              textTransform: "uppercase", letterSpacing: "0.04em",
                            }}>
                              {sev}
                            </span>
                          ) : <span style={{ color: "var(--text-faint)" }}>—</span>}
                        </td>
                        <td style={{ padding: "0.75rem 0.85rem", color: "var(--text-faint)", fontSize: "0.75rem", background: "var(--bg-surface)" }}>{det.model_version}</td>
                        <td style={{ padding: "0.75rem 0.85rem", color: "var(--text-faint)", fontSize: "0.75rem", background: "var(--bg-surface)" }}>{formatDate(det.created_at)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {/* Pagination */}
              <div style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "0.75rem 1rem", borderTop: "1px solid var(--border)",
              }}>
                <span style={{ color: "var(--text-faint)", fontSize: "0.75rem" }}>
                  {skip + 1}–{skip + detections.length}
                </span>
                <div style={{ display: "flex", gap: "0.5rem" }}>
                  <button className="page-btn" onClick={() => setSkip(Math.max(0, skip - LIMIT))} disabled={skip === 0}>← Prev</button>
                  <button className="page-btn" onClick={() => setSkip(skip + LIMIT)} disabled={detections.length < LIMIT}>Next →</button>
                </div>
              </div>
            </div>

            {/* ── Mobile Cards ── */}
            <div className="cards-wrapper">
              {detections.map((det) => {
                const dmgTypes = getDamageTypes(det.results);
                const sev      = getOverallSev(det.results);
                const sevColor = SEVERITY_COLORS[sev] || "#64748b";
                return (
                  <div key={det.id} className="history-card">
                    {/* Top row: ID + severity */}
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ color: "var(--text-faint)", fontSize: "0.75rem" }}>#{det.id}</span>
                      {sev && (
                        <span style={{
                          background: sevColor + "18", color: sevColor,
                          border: `1px solid ${sevColor}44`, borderRadius: "6px",
                          padding: "1px 8px", fontSize: "0.68rem", fontWeight: "700",
                          textTransform: "uppercase",
                        }}>
                          {sev}
                        </span>
                      )}
                    </div>
                    {/* Filename */}
                    <div>
                      <p className="history-row-label">Image</p>
                      <p style={{ color: "var(--text-secondary)", fontSize: "0.8rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {det.image_filename}
                      </p>
                    </div>
                    {/* Damage types */}
                    <div>
                      <p className="history-row-label">Damage Types</p>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "3px", marginTop: "2px" }}>
                        {dmgTypes.length > 0
                          ? dmgTypes.map((cls) => (
                              <span key={cls} style={{
                                background: (CLASS_COLORS[cls] || "#64748b") + "18",
                                color: CLASS_COLORS[cls] || "#64748b",
                                border: `1px solid ${(CLASS_COLORS[cls] || "#64748b")}33`,
                                borderRadius: "6px", padding: "1px 6px",
                                fontSize: "0.7rem", fontWeight: "500",
                              }}>
                                {CLASS_LABELS[cls] || cls.replace(/_/g, " ")}
                              </span>
                            ))
                          : <span style={{ color: "var(--text-faint)", fontSize: "0.75rem" }}>None</span>
                        }
                      </div>
                    </div>
                    {/* Bottom row: instances + date */}
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{
                        background: "var(--accent)22", color: "var(--accent)",
                        border: "1px solid var(--accent)44", borderRadius: "6px",
                        padding: "1px 7px", fontSize: "0.72rem", fontWeight: "600",
                      }}>
                        {det.total_instances} instance{det.total_instances !== 1 ? "s" : ""}
                      </span>
                      <span style={{ color: "var(--text-faint)", fontSize: "0.72rem" }}>
                        {formatDate(det.created_at)}
                      </span>
                    </div>
                  </div>
                );
              })}

              {/* Mobile pagination */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.25rem 0" }}>
                <span style={{ color: "var(--text-faint)", fontSize: "0.75rem" }}>{skip + 1}–{skip + detections.length}</span>
                <div style={{ display: "flex", gap: "0.5rem" }}>
                  <button className="page-btn" onClick={() => setSkip(Math.max(0, skip - LIMIT))} disabled={skip === 0}>← Prev</button>
                  <button className="page-btn" onClick={() => setSkip(skip + LIMIT)} disabled={detections.length < LIMIT}>Next →</button>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}