const CLASS_COLORS = {
  dent: "#ff6347", scratch: "#3b82f6", crack: "#22c55e",
  glass_shatter: "#a855f7", lamp_broken: "#eab308", tire_flat: "#f97316",
};
const CLASS_LABELS = {
  dent: "Dent", scratch: "Scratch", crack: "Crack",
  glass_shatter: "Glass Shatter", lamp_broken: "Lamp Broken", tire_flat: "Tire Flat",
};

function ConfidenceBar({ value, color }) {
  return (
    <div style={{ width: "48px", height: "4px", borderRadius: "2px", background: "var(--bg-elevated)", overflow: "hidden" }}>
      <div style={{ width: `${(value * 100).toFixed(0)}%`, height: "100%", background: color, borderRadius: "2px", transition: "width 0.4s ease" }} />
    </div>
  );
}

export default function ResultCard({ result }) {
  const isRejected = !result.verified;
  const hasDamage  = result.detections?.length > 0;

  return (
    <>
      <style>{`
        .result-card {
          background: var(--bg-surface);
          border-radius: 14px;
          border: 1px solid var(--border);
          overflow: hidden;
          transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
        }
        .result-card:hover { border-color: var(--text-faint); box-shadow: var(--shadow); }
        .result-card.rejected { border-color: var(--danger); opacity: 0.85; }
        .detection-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: var(--bg-elevated);
          border-radius: 8px;
          padding: 0.5rem 0.6rem;
          border: 1px solid var(--border);
          gap: 0.5rem;
        }
        .det-left {
          display: flex; align-items: center; gap: 0.4rem;
          min-width: 0; flex: 1;
        }
        .det-right {
          display: flex; flex-direction: column;
          align-items: flex-end; gap: 2px;
          flex-shrink: 0;
        }
        @media (max-width: 400px) {
          .detection-row { flex-direction: column; align-items: flex-start; gap: 0.4rem; }
          .det-right { flex-direction: row; align-items: center; gap: 0.5rem; }
        }
      `}</style>

      <div className={`result-card ${isRejected ? "rejected" : ""}`}>

        {/* Image */}
        <div style={{ position: "relative" }}>
          <img
            src={`data:image/jpeg;base64,${result.annotated_image}`}
            alt={result.filename}
            style={{
              width: "100%", display: "block",
              maxHeight: "260px", objectFit: "contain",
              background: "#000", opacity: isRejected ? 0.4 : 1,
            }}
          />
          {/* Status */}
          <div style={{
            position: "absolute", top: "8px", left: "8px",
            background: isRejected ? "#ef444466" : "#22c55e66",
            backdropFilter: "blur(8px)", borderRadius: "8px",
            padding: "2px 8px", fontSize: "0.7rem",
            fontWeight: "600", color: "white",
          }}>
            {isRejected ? "✕ Not a car" : "✓ Verified"}
          </div>
          {/* Instance count */}
          {!isRejected && (
            <div style={{
              position: "absolute", top: "8px", right: "8px",
              background: "#3b82f666", backdropFilter: "blur(8px)",
              borderRadius: "8px", padding: "2px 8px",
              fontSize: "0.7rem", fontWeight: "600", color: "white",
            }}>
              {result.total_instances} instance{result.total_instances !== 1 ? "s" : ""}
            </div>
          )}
          {/* Overall severity */}
          {!isRejected && result.overall_severity?.label && result.overall_severity.label !== "None" && (
            <div style={{
              position: "absolute", bottom: "8px", right: "8px",
              background: result.overall_severity.color + "bb",
              backdropFilter: "blur(8px)", borderRadius: "8px",
              padding: "3px 10px", fontSize: "0.72rem",
              fontWeight: "700", color: "white",
              letterSpacing: "0.04em", textTransform: "uppercase",
            }}>
              {result.overall_severity.label}
            </div>
          )}
        </div>

        {/* Body */}
        <div style={{ padding: "0.85rem" }}>
          <p style={{
            fontSize: "0.75rem", color: "var(--text-muted)",
            marginBottom: "0.65rem", overflow: "hidden",
            textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {result.filename}
          </p>

          {isRejected && (
            <p style={{
              color: "var(--danger)", fontSize: "0.8rem", textAlign: "center",
              padding: "0.65rem", background: "var(--bg-elevated)", borderRadius: "8px",
            }}>
              {result.rejection_reason || "No car detected in this image"}
            </p>
          )}

          {!isRejected && (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
              {hasDamage ? result.detections.map((det, i) => {
                const color = CLASS_COLORS[det.class_name] || "#64748b";
                return (
                  <div key={i} className="detection-row" style={{ borderLeft: `3px solid ${color}` }}>
                    <div className="det-left">
                      <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: color, flexShrink: 0 }} />
                      <span style={{ fontSize: "0.8rem", color: "var(--text-primary)", fontWeight: "500", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {CLASS_LABELS[det.class_name] || det.class_name.replace(/_/g, " ")}
                      </span>
                    </div>
                    <div className="det-right">
                      <span style={{ fontSize: "0.62rem", color: "var(--text-faint)", letterSpacing: "0.04em" }}>
                        CONFIDENCE
                      </span>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                        <ConfidenceBar value={det.confidence} color={color} />
                        <span style={{ fontSize: "0.73rem", fontWeight: "600", color, minWidth: "30px", textAlign: "right" }}>
                          {(det.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                );
              }) : (
                <div style={{
                  textAlign: "center", padding: "0.85rem",
                  background: "var(--bg-elevated)", borderRadius: "8px",
                  border: "1px solid var(--border)", color: "var(--success)",
                  fontSize: "0.8rem", fontWeight: "500",
                }}>
                  No damage detected
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}