import { useState } from "react";
import UploadZone from "../components/UploadZone";
import ResultCard from "../components/ResultCard";
import { detectDamage } from "../services/api";

const CLASS_COLORS = {
  dent: "#ff6347", scratch: "#3b82f6", crack: "#22c55e",
  glass_shatter: "#a855f7", lamp_broken: "#eab308", tire_flat: "#f97316",
};
const CLASS_LABELS = {
  dent: "Dent", scratch: "Scratch", crack: "Crack",
  glass_shatter: "Glass Shatter", lamp_broken: "Lamp Broken", tire_flat: "Tire Flat",
};

function SeverityBar({ score }) {
  const pct   = Math.round(score * 100);
  const color = score < 0.35 ? "var(--success)" : score < 0.65 ? "var(--warning)" : "var(--danger)";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
      <div style={{ flex: 1, height: "4px", background: "var(--bg-base)", borderRadius: "4px", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: "4px", transition: "width 0.4s ease" }} />
      </div>
      <span style={{ fontSize: "0.72rem", color, fontWeight: "600", minWidth: "30px", textAlign: "right" }}>{pct}%</span>
    </div>
  );
}

function AggregatedDamageCard({ group, groupIndex }) {
  return (
    <div style={{
      background: "var(--bg-surface)", border: "1px solid var(--border)",
      borderRadius: "12px", padding: "1rem", marginBottom: "1.25rem",
      boxShadow: "var(--shadow)",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.85rem", gap: "0.5rem" }}>
        <div>
          <h3 style={{ color: "var(--text-primary)", fontSize: "0.92rem", fontWeight: "700" }}>
            Car {groupIndex + 1}
            <span style={{ color: "var(--text-faint)", fontWeight: "400", marginLeft: "0.4rem", fontSize: "0.78rem" }}>
              {group.image_count} angle{group.image_count > 1 ? "s" : ""}
            </span>
          </h3>
          <p style={{ color: "var(--text-muted)", fontSize: "0.75rem", marginTop: "2px" }}>
            {group.total_instances} instance{group.total_instances !== 1 ? "s" : ""} detected
          </p>
        </div>
        {group.overall_severity?.label && group.overall_severity.label !== "None" && (
          <span style={{
            background: group.overall_severity.color + "18",
            color: group.overall_severity.color,
            border: `1px solid ${group.overall_severity.color}44`,
            borderRadius: "6px", padding: "3px 10px",
            fontSize: "0.7rem", fontWeight: "700",
            letterSpacing: "0.05em", textTransform: "uppercase",
            flexShrink: 0,
          }}>
            {group.overall_severity.label}
          </span>
        )}
      </div>

      {group.aggregated_damage?.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
          {group.aggregated_damage.map((dmg, i) => (
            <div key={i} style={{
              background: "var(--bg-elevated)", borderRadius: "8px",
              padding: "0.5rem 0.65rem",
              borderLeft: `3px solid ${CLASS_COLORS[dmg.class_name] || "#64748b"}`,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.3rem", gap: "0.5rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", minWidth: 0 }}>
                  <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: CLASS_COLORS[dmg.class_name] || "#64748b", flexShrink: 0 }} />
                  <span style={{ fontSize: "0.8rem", color: "var(--text-primary)", fontWeight: "500", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {CLASS_LABELS[dmg.class_name] || dmg.class_name.replace(/_/g, " ")}
                  </span>
                  {dmg.count > 1 && <span style={{ color: "var(--text-faint)", fontSize: "0.7rem", flexShrink: 0 }}>×{dmg.count}</span>}
                </div>
                <span style={{
                  background: dmg.severity_color + "18", color: dmg.severity_color,
                  border: `1px solid ${dmg.severity_color}44`, borderRadius: "6px",
                  padding: "1px 7px", fontSize: "0.68rem", fontWeight: "700",
                  textTransform: "uppercase", letterSpacing: "0.04em", flexShrink: 0,
                }}>
                  {dmg.severity_label}
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span style={{ fontSize: "0.62rem", color: "var(--text-faint)", letterSpacing: "0.04em", whiteSpace: "nowrap" }}>
                  SEVERITY SCORE
                </span>
                <SeverityBar score={dmg.severity_score} />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div style={{
          textAlign: "center", padding: "0.65rem",
          background: "var(--bg-elevated)", borderRadius: "8px",
          border: "1px solid var(--border)", color: "var(--success)",
          fontSize: "0.8rem", fontWeight: "500",
        }}>
          No damage detected across all angles
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [files,   setFiles]   = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const handleDetect = async () => {
    if (!files.length) return;
    setLoading(true); setError(null); setResults(null);
    try {
      const data = await detectDamage(files);
      setResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => { setFiles([]); setResults(null); setError(null); };
  const isDisabled  = !files.length || loading;

  return (
    <>
      <style>{`
        .detect-btn {
          background: linear-gradient(135deg, var(--accent), #0ea5e9);
          color: white; border: none; border-radius: 8px;
          padding: 0.65rem 1.5rem; font-size: 0.88rem; font-weight: 600;
          cursor: pointer; transition: opacity 0.2s, transform 0.15s;
          width: 100%;
        }
        .detect-btn:hover:not(:disabled) { opacity: 0.9; transform: translateY(-1px); }
        .detect-btn:disabled { background: var(--bg-elevated); color: var(--text-faint); cursor: not-allowed; transform: none; }
        .reset-btn {
          background: transparent; color: var(--text-muted);
          border: 1px solid var(--border); border-radius: 8px;
          padding: 0.55rem 1rem; cursor: pointer; font-size: 0.82rem;
          transition: all 0.2s; white-space: nowrap;
        }
        .reset-btn:hover { border-color: var(--text-faint); color: var(--text-secondary); }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        .dot { display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--accent); animation:pulse 1.2s ease-in-out infinite; }
        .dot:nth-child(2){animation-delay:0.2s}
        .dot:nth-child(3){animation-delay:0.4s}
        .results-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: 0.85rem;
          margin-bottom: 1.75rem;
        }
        @media (max-width: 400px) {
          .results-grid { grid-template-columns: 1fr; }
        }
        .summary-bar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.25rem;
          flex-wrap: wrap;
          gap: 0.75rem;
        }
      `}</style>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "1rem" }}>

        <div style={{ marginBottom: "1.5rem" }}>
          <h1 style={{ fontSize: "1.4rem", fontWeight: "700", color: "var(--text-primary)" }}>
            Detect Car Damage
          </h1>
          <p style={{ color: "var(--text-muted)", marginTop: "0.3rem", fontSize: "0.85rem" }}>
            Upload up to 10 car images to detect and classify damage
          </p>
        </div>

        {!results && (
          <>
            <UploadZone onFilesSelected={setFiles} />
            {error && (
              <div style={{
                background: "var(--bg-elevated)", border: "1px solid var(--danger)",
                borderRadius: "8px", padding: "0.75rem",
                marginTop: "0.85rem", color: "var(--danger)", fontSize: "0.82rem",
              }}>
                {error}
              </div>
            )}
            <div style={{ marginTop: "1rem" }}>
              <button className="detect-btn" onClick={handleDetect} disabled={isDisabled}>
                {loading ? "Detecting..." : files.length > 0
                  ? `Detect Damage — ${files.length} image${files.length > 1 ? "s" : ""}`
                  : "Detect Damage"}
              </button>
            </div>
          </>
        )}

        {loading && (
          <div style={{ marginTop: "2rem", textAlign: "center", padding: "1.5rem" }}>
            <div style={{ display: "flex", justifyContent: "center", gap: "6px", marginBottom: "0.85rem" }}>
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
            <p style={{ fontSize: "0.88rem", color: "var(--text-muted)" }}>
              Running pipeline on {files.length} image{files.length > 1 ? "s" : ""}
            </p>
            <p style={{ fontSize: "0.75rem", color: "var(--text-faint)", marginTop: "0.35rem" }}>
              Verifying → Grouping → Detecting → Scoring
            </p>
          </div>
        )}

        {results && (
          <div>
            <div className="summary-bar">
              <div>
                <h2 style={{ fontSize: "1.1rem", fontWeight: "700", color: "var(--text-primary)" }}>
                  Detection Complete
                </h2>
                <p style={{ color: "var(--text-muted)", fontSize: "0.78rem", marginTop: "0.2rem" }}>
                  {results.total_images} uploaded · {results.verified_images} verified · {results.total_instances} instances
                </p>
              </div>
              <button className="reset-btn" onClick={handleReset}>← New</button>
            </div>

            {results.rejected_images?.length > 0 && (
              <div style={{
                background: "var(--bg-elevated)", border: "1px solid var(--danger)",
                borderRadius: "10px", padding: "0.85rem", marginBottom: "1.25rem",
              }}>
                <p style={{ color: "var(--danger)", fontWeight: "600", fontSize: "0.82rem", marginBottom: "0.4rem" }}>
                  {results.rejected_images.length} image{results.rejected_images.length > 1 ? "s" : ""} rejected
                </p>
                {results.rejected_images.map((r, i) => (
                  <p key={i} style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>• {r.filename} — {r.reason}</p>
                ))}
              </div>
            )}

            {results.car_groups?.length > 0
              ? results.car_groups.map((group, gi) => (
                  <div key={gi}>
                    <AggregatedDamageCard group={group} groupIndex={gi} />
                    <div className="results-grid">
                      {group.images.map((img, i) => <ResultCard key={i} result={img} />)}
                    </div>
                  </div>
                ))
              : (
                <div style={{ textAlign: "center", padding: "2.5rem", color: "var(--text-faint)" }}>
                  <p style={{ fontSize: "2rem", marginBottom: "0.6rem" }}>🚗</p>
                  <p style={{ fontSize: "0.88rem" }}>No car images detected.</p>
                </div>
              )
            }
          </div>
        )}
      </div>
    </>
  );
}