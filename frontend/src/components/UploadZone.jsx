import { useDropzone } from "react-dropzone";
import { useState } from "react";

export default function UploadZone({ onFilesSelected }) {
  const [previews, setPreviews] = useState([]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept:   { "image/*": [".jpg", ".jpeg", ".png"] },
    maxFiles: 10,
    onDrop:   (acceptedFiles) => {
      const newPreviews = acceptedFiles.map((file) => ({
        file, url: URL.createObjectURL(file),
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
    <>
      <style>{`
        .dropzone {
          border: 2px dashed var(--border);
          border-radius: 16px;
          padding: 2rem 1rem;
          text-align: center;
          cursor: pointer;
          background: var(--bg-surface);
          transition: all 0.25s ease;
        }
        .dropzone:hover { border-color: var(--accent); }
        .dropzone.active { border-color: var(--accent); background: var(--bg-elevated); }
        .dropzone-icon {
          width: 52px; height: 52px;
          border-radius: 14px;
          background: var(--bg-elevated);
          border: 1px solid var(--border);
          display: flex; align-items: center; justify-content: center;
          margin: 0 auto 0.75rem;
          transition: transform 0.2s;
        }
        .dropzone:hover .dropzone-icon { transform: scale(1.05); }
        .preview-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
          gap: 0.6rem;
          margin-top: 1rem;
        }
        @media (max-width: 400px) {
          .preview-grid { grid-template-columns: repeat(auto-fill, minmax(72px, 1fr)); }
        }
        .preview-img {
          width: 100%; height: 80px;
          object-fit: cover; border-radius: 8px;
          border: 1px solid var(--border); display: block;
          transition: border-color 0.2s;
        }
        .preview-img:hover { border-color: var(--accent); }
        .remove-btn {
          position: absolute; top: 4px; right: 4px;
          background: #ef444499; border: none; border-radius: 50%;
          width: 20px; height: 20px; cursor: pointer; color: white;
          font-size: 13px; line-height: 1;
          display: flex; align-items: center; justify-content: center;
          transition: background 0.2s;
        }
        .remove-btn:hover { background: #ef4444; }
      `}</style>

      <div {...getRootProps()} className={`dropzone ${isDragActive ? "active" : ""}`}>
        <input {...getInputProps()} />
        <div className="dropzone-icon">
          {isDragActive
            ? <span style={{ fontSize: "1.4rem" }}>⬇️</span>
            : <img src="/camerawowo.png" alt="upload" style={{ width: "28px", height: "28px", objectFit: "contain" }} />
          }
        </div>
        <p style={{ color: "var(--text-primary)", fontSize: "0.95rem", fontWeight: "500", marginBottom: "0.3rem" }}>
          {isDragActive ? "Release to upload" : "Drop car images here"}
        </p>
        <p style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>
          or tap to browse · JPG, PNG · up to 10
        </p>
        {previews.length > 0 && (
          <div style={{
            display: "inline-flex", alignItems: "center", gap: "0.4rem",
            marginTop: "0.75rem", background: "var(--bg-elevated)",
            border: "1px solid var(--border)", borderRadius: "20px",
            padding: "3px 12px", fontSize: "0.75rem",
            color: "var(--accent)", fontWeight: "600",
          }}>
            ✓ {previews.length} image{previews.length !== 1 ? "s" : ""} selected
          </div>
        )}
      </div>

      {previews.length > 0 && (
        <div className="preview-grid">
          {previews.map((preview, index) => (
            <div key={index} style={{ position: "relative" }}>
              <img src={preview.url} alt={`preview-${index}`} className="preview-img" />
              <button className="remove-btn" onClick={(e) => { e.stopPropagation(); removeFile(index); }}>×</button>
              <p style={{
                fontSize: "0.65rem", color: "var(--text-faint)",
                marginTop: "0.25rem", overflow: "hidden",
                textOverflow: "ellipsis", whiteSpace: "nowrap",
              }}>
                {preview.file.name}
              </p>
            </div>
          ))}
        </div>
      )}
    </>
  );
}