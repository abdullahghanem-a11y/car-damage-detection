import { useState } from "react";
import UploadZone from "../components/UploadZone";
import ResultCard from "../components/ResultCard";
import { detectDamage } from "../services/api";

const CLASS_EMOJIS = {
  dent:          "🔴",
  scratch:       "🔵",
  crack:         "🟢",
  glass_shatter: "🟣",
  lamp_broken:   "🟡",
  tire_flat:     "🟠",
};

function SeverityBar({ score }) {
  const pct   = Math.round(score * 100);
  const color = score < 0.35 ? "#22c55e" : score < 0.65 ? "#f59e0b" : "#ef4444";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
      <div style={{
        flex:         1,
        height:       "6px",
        background:   "#0f172a",
        borderRadius: "4px",
        overflow:     "hidden",
      }}>
        <div style={{
          width:        `${pct}%`,
          height:       "100%",
          background:   color,
          borderRadius: "4px",
          transition:   "width 0.4s ease",
        }} />
      </div>
      <span style={{ fontSize: "0.8rem", color, fontWeight: "600", minWidth: "36px" }}>
        {pct}%
      </span>
    </div>
  );
}

function AggregatedDamageCard({ group, groupIndex }) {
  return (
    <div style={{
      background:   "#1e293b",
      border:       "1px solid #334155",
      borderRadius: "12px",
      padding:      "1.25rem",
      marginBottom: "2rem",
    }}>
      {/* Group header */}
      <div style={{
        display:        "flex",
        justifyContent: "space-between",
        alignItems:     "center",
        marginBottom:   "1rem",
      }}>
        <div>
          <h3 style={{ color: "#f1f5f9", fontSize: "1rem", fontWeight: "700" }}>
            🚗 Car {groupIndex + 1} — {group.image_count} angle{group.image_count > 1 ? "s" : ""}
          </h3>
          <p style={{ color: "#64748b", fontSize: "0.8rem", marginTop: "2px" }}>
            {group.total_instances} total damage instance{group.total_instances !== 1 ? "s" : ""} detected
          </p>
        </div>
        {group.overall_severity && group.overall_severity.label !== "None" && (
          <span style={{
            background:   group.overall_severity.color + "22",
            color:        group.overall_severity.color,
            border:       `1px solid ${group.overall_severity.color}66`,
            borderRadius: "20px",
            padding:      "4px 14px",
            fontSize:     "0.85rem",
            fontWeight:   "700",
          }}>
            {group.overall_severity.label} Overall
          </span>
        )}
      </div>

      {/* Aggregated damage types */}
      {group.aggregated_damage && group.aggregated_damage.length > 0 && (
        <div>
          <p style={{ color: "#64748b", fontSize: "0.75rem", marginBottom: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Aggregated Damage Summary
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            {group.aggregated_damage.map((dmg, i) => (
              <div key={i} style={{
                background:   "#0f172a",
                borderRadius: "8px",
                padding:      "0.6rem 0.75rem",
              }}>
                <div style={{
                  display:        "flex",
                  justifyContent: "space-between",
                  alignItems:     "center",
                  marginBottom:   "0.4rem",
                }}>
                  <span style={{ fontSize: "0.85rem", color: "#e2e8f0" }}>
                    {CLASS_EMOJIS[dmg.class_name]} {dmg.class_name.replace(/_/g, " ")}
                    {dmg.count > 1 && (
                      <span style={{ color: "#64748b", fontSize: "0.75rem", marginLeft: "6px" }}>
                        ×{dmg.count} angles
                      </span>
                    )}
                  </span>
                  <span style={{
                    background:   dmg.severity_color + "22",
                    color:        dmg.severity_color,
                    border:       `1px solid ${dmg.severity_color}66`,
                    borderRadius: "12px",
                    padding:      "2px 10px",
                    fontSize:     "0.75rem",
                    fontWeight:   "700",
                  }}>
                    {dmg.severity_label}
                  </span>
                </div>
                <SeverityBar score={dmg.severity_score} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No damage */}
      {group.aggregated_damage && group.aggregated_damage.length === 0 && (
        <p style={{ color: "#22c55e", fontSize: "0.85rem", textAlign: "center", padding: "0.5rem" }}>
          ✅ No damage detected across all angles
        </p>
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
      {!results && <UploadZone onFilesSelected={setFiles} />}

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
          ❌ {error}
        </div>
      )}

      {/* Detect Button */}
      {!results && (
        <div style={{ marginTop: "1.5rem" }}>
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
                ? `Detect Damage (${files.length} image${files.length > 1 ? "s" : ""})`
                : "Detect Damage"}
          </button>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div style={{ marginTop: "2rem", textAlign: "center", color: "#64748b" }}>
          <p>Running pipeline on {files.length} image{files.length > 1 ? "s" : ""}...</p>
          <p style={{ fontSize: "0.8rem", marginTop: "0.5rem" }}>
            Verifying cars → Grouping angles → Detecting damage → Scoring severity
          </p>
        </div>
      )}

      {/* Results */}
      {results && (
        <div>

          {/* Top summary bar */}
          <div style={{
            display:        "flex",
            justifyContent: "space-between",
            alignItems:     "center",
            marginBottom:   "1.5rem",
            flexWrap:       "wrap",
            gap:            "1rem",
          }}>
            <div>
              <h2 style={{ fontSize: "1.3rem", fontWeight: "700", color: "#f1f5f9" }}>
                Detection Complete
              </h2>
              <p style={{ color: "#64748b", fontSize: "0.85rem", marginTop: "0.25rem" }}>
                {results.total_images} uploaded — {results.verified_images} verified — {results.total_instances} damage instances
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

          {/* Rejected images */}
          {results.rejected_images && results.rejected_images.length > 0 && (
            <div style={{
              background:   "#450a0a22",
              border:       "1px solid #ef444444",
              borderRadius: "10px",
              padding:      "1rem",
              marginBottom: "1.5rem",
            }}>
              <p style={{ color: "#ef4444", fontWeight: "600", fontSize: "0.9rem", marginBottom: "0.5rem" }}>
                ❌ {results.rejected_images.length} image{results.rejected_images.length > 1 ? "s" : ""} rejected — no car detected
              </p>
              {results.rejected_images.map((r, i) => (
                <p key={i} style={{ color: "#fca5a5", fontSize: "0.8rem" }}>
                  • {r.filename} — {r.reason}
                </p>
              ))}
            </div>
          )}

          {/* Aggregated summary per car group */}
          {results.car_groups && results.car_groups.length > 0 && (
            <div>
              {results.car_groups.map((group, groupIndex) => (
                <div key={groupIndex}>
                  {/* Aggregated damage card */}
                  <AggregatedDamageCard group={group} groupIndex={groupIndex} />

                  {/* Per-image result cards */}
                  <div style={{
                    display:             "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
                    gap:                 "1.25rem",
                    marginBottom:        "2.5rem",
                  }}>
                    {group.images.map((img, i) => (
                      <ResultCard key={i} result={img} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* No verified images */}
          {results.car_groups && results.car_groups.length === 0 && (
            <div style={{ textAlign: "center", padding: "3rem", color: "#64748b" }}>
              <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🚗</div>
              <p>No car images detected. Please upload images that contain cars.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}