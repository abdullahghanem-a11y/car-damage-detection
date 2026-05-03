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

export default function ResultCard({ result }) {
  return (
    <div style={{
      background:   "#1e293b",
      borderRadius: "12px",
      border:       "1px solid #334155",
      overflow:     "hidden",
    }}>
      {/* Annotated Image */}
      <div style={{ position: "relative" }}>
        <img
          src={`data:image/jpeg;base64,${result.annotated_image}`}
          alt={result.filename}
          style={{ width: "100%", display: "block", maxHeight: "400px", objectFit: "contain" }}
        />
        {/* Instance badge */}
        <div style={{
          position:     "absolute",
          top:          "12px",
          right:        "12px",
          background:   "#3b82f6",
          borderRadius: "20px",
          padding:      "4px 12px",
          fontSize:     "0.8rem",
          fontWeight:   "600",
          color:        "white",
        }}>
          {result.total_instances} instance{result.total_instances !== 1 ? "s" : ""}
        </div>
      </div>

      {/* Details */}
      <div style={{ padding: "1.25rem" }}>
        <p style={{
          fontSize:     "0.9rem",
          fontWeight:   "600",
          color:        "#f1f5f9",
          marginBottom: "1rem",
          overflow:     "hidden",
          textOverflow: "ellipsis",
          whiteSpace:   "nowrap",
        }}>
          📄 {result.filename}
        </p>

        {/* Detections */}
        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
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
              <span style={{ fontSize: "0.85rem", color: "#e2e8f0" }}>
                {CLASS_EMOJIS[det.class_name]} {det.class_name.replace("_", " ")}
              </span>
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
          ))}
        </div>

        {/* No detections */}
        {result.detections.length === 0 && (
          <p style={{ color: "#64748b", fontSize: "0.85rem", textAlign: "center" }}>
            ✅ No damage detected
          </p>
        )}
      </div>
    </div>
  );
}