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

function SeverityBadge({ label, color }) {
  if (!label || label === "None") return null;
  return (
    <span style={{
      background:   color + "22",
      color:        color,
      border:       `1px solid ${color}66`,
      borderRadius: "12px",
      padding:      "2px 10px",
      fontSize:     "0.75rem",
      fontWeight:   "700",
      letterSpacing:"0.03em",
    }}>
      {label}
    </span>
  );
}

export default function ResultCard({ result }) {
  const isRejected = !result.verified;

  return (
    <div style={{
      background:   "#1e293b",
      borderRadius: "12px",
      border:       `1px solid ${isRejected ? "#ef444444" : "#334155"}`,
      overflow:     "hidden",
    }}>

      {/* Annotated Image */}
      <div style={{ position: "relative" }}>
        <img
          src={`data:image/jpeg;base64,${result.annotated_image}`}
          alt={result.filename}
          style={{
            width:      "100%",
            display:    "block",
            maxHeight:  "300px",
            objectFit:  "contain",
            opacity:    isRejected ? 0.5 : 1,
          }}
        />

        {/* Verified / Rejected badge */}
        <div style={{
          position:     "absolute",
          top:          "10px",
          left:         "10px",
          background:   isRejected ? "#ef444488" : "#22c55e88",
          backdropFilter: "blur(4px)",
          borderRadius: "20px",
          padding:      "3px 10px",
          fontSize:     "0.75rem",
          fontWeight:   "600",
          color:        "white",
        }}>
          {isRejected ? "❌ Not a car" : "✅ Car verified"}
        </div>

        {/* Instance count badge */}
        {!isRejected && (
          <div style={{
            position:     "absolute",
            top:          "10px",
            right:        "10px",
            background:   "#3b82f688",
            backdropFilter: "blur(4px)",
            borderRadius: "20px",
            padding:      "3px 10px",
            fontSize:     "0.75rem",
            fontWeight:   "600",
            color:        "white",
          }}>
            {result.total_instances} instance{result.total_instances !== 1 ? "s" : ""}
          </div>
        )}

        {/* Overall severity badge */}
        {!isRejected && result.overall_severity && result.overall_severity.label !== "None" && (
          <div style={{
            position:       "absolute",
            bottom:         "10px",
            right:          "10px",
            background:     result.overall_severity.color + "cc",
            backdropFilter: "blur(4px)",
            borderRadius:   "20px",
            padding:        "3px 12px",
            fontSize:       "0.8rem",
            fontWeight:     "700",
            color:          "white",
          }}>
            {result.overall_severity.label} Damage
          </div>
        )}
      </div>

      {/* Details */}
      <div style={{ padding: "1rem" }}>

        {/* Filename */}
        <p style={{
          fontSize:     "0.85rem",
          fontWeight:   "600",
          color:        "#f1f5f9",
          marginBottom: "0.75rem",
          overflow:     "hidden",
          textOverflow: "ellipsis",
          whiteSpace:   "nowrap",
        }}>
          📄 {result.filename}
        </p>

        {/* Rejected reason */}
        {isRejected && (
          <p style={{
            color:     "#ef4444",
            fontSize:  "0.85rem",
            textAlign: "center",
            padding:   "0.5rem",
          }}>
            ⚠️ {result.rejection_reason || "No car detected in this image"}
          </p>
        )}

        {/* Detections */}
        {!isRejected && (
          <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
            {result.detections.map((det, i) => (
              <div key={i} style={{
                display:        "flex",
                justifyContent: "space-between",
                alignItems:     "center",
                background:     "#0f172a",
                borderRadius:   "8px",
                padding:        "0.5rem 0.75rem",
                borderLeft:     `3px solid ${CLASS_COLORS[det.class_name] || "#64748b"}`,
              }}>
                {/* Class name */}
                <span style={{ fontSize: "0.85rem", color: "#e2e8f0" }}>
                  {CLASS_EMOJIS[det.class_name]} {det.class_name.replace(/_/g, " ")}
                </span>

                {/* Right side: severity + confidence */}
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <SeverityBadge
                    label={det.severity_label}
                    color={det.severity_color}
                  />
                  <span style={{
                    fontSize:     "0.8rem",
                    fontWeight:   "600",
                    color:        CLASS_COLORS[det.class_name] || "#64748b",
                    background:   "#1e293b",
                    padding:      "2px 8px",
                    borderRadius: "12px",
                  }}>
                    {(det.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}

            {/* No detections */}
            {result.detections.length === 0 && (
              <p style={{ color: "#22c55e", fontSize: "0.85rem", textAlign: "center", padding: "0.5rem" }}>
                ✅ No damage detected
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}