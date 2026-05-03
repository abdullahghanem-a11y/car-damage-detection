import { useDropzone } from "react-dropzone";
import { useState } from "react";

export default function UploadZone({ onFilesSelected }) {
  const [previews, setPreviews] = useState([]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept:   { "image/*": [".jpg", ".jpeg", ".png"] },
    maxFiles: 10,
    onDrop:   (acceptedFiles) => {
      const newPreviews = acceptedFiles.map((file) => ({
        file,
        url: URL.createObjectURL(file),
      }));
      setPreviews(newPreviews);
      onFilesSelected(acceptedFiles);
    },
  });

  const removeFile = (index) => {
    const updated = previews.filter((_, i) => i !== index);
    setPreviews(updated);
    onFilesSelected(updated.map((p) => p.file));
  };

  return (
    <div>
      {/* Drop Zone */}
      <div
        {...getRootProps()}
        style={{
          border:        `2px dashed ${isDragActive ? "#3b82f6" : "#334155"}`,
          borderRadius:  "12px",
          padding:       "3rem",
          textAlign:     "center",
          cursor:        "pointer",
          background:    isDragActive ? "#1e3a5f" : "#1e293b",
          transition:    "all 0.2s ease",
        }}
      >
        <input {...getInputProps()} />
        <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>📸</div>
        <p style={{ color: "#94a3b8", fontSize: "1rem" }}>
          {isDragActive
            ? "Drop images here..."
            : "Drag & drop car images here, or click to select"}
        </p>
        <p style={{ color: "#475569", fontSize: "0.85rem", marginTop: "0.5rem" }}>
          Supports JPG, JPEG, PNG — up to 10 images
        </p>
      </div>

      {/* Previews */}
      {previews.length > 0 && (
        <div style={{
          display:             "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))",
          gap:                 "1rem",
          marginTop:           "1.5rem",
        }}>
          {previews.map((preview, index) => (
            <div key={index} style={{ position: "relative" }}>
              <img
                src={preview.url}
                alt={`preview-${index}`}
                style={{
                  width:        "100%",
                  height:       "100px",
                  objectFit:    "cover",
                  borderRadius: "8px",
                  border:       "1px solid #334155",
                }}
              />
              <button
                onClick={(e) => { e.stopPropagation(); removeFile(index); }}
                style={{
                  position:     "absolute",
                  top:          "4px",
                  right:        "4px",
                  background:   "#ef4444",
                  border:       "none",
                  borderRadius: "50%",
                  width:        "20px",
                  height:       "20px",
                  cursor:       "pointer",
                  color:        "white",
                  fontSize:     "12px",
                  display:      "flex",
                  alignItems:   "center",
                  justifyContent: "center",
                }}
              >
                ×
              </button>
              <p style={{
                fontSize:     "0.7rem",
                color:        "#64748b",
                marginTop:    "0.25rem",
                overflow:     "hidden",
                textOverflow: "ellipsis",
                whiteSpace:   "nowrap",
              }}>
                {preview.file.name}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}