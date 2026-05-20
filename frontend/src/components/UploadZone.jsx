import { useDropzone } from "react-dropzone";
import { useState, useRef } from "react";

export default function UploadZone({ onFilesSelected }) {
  const [previews,    setPreviews]    = useState([]);
  const [showPicker,  setShowPicker]  = useState(false);

  const cameraInputRef  = useRef(null);
  const galleryInputRef = useRef(null);

  // ── Shared handler for both sources ───────────────────────────────────
  const handleFiles = (files) => {
    const accepted = Array.from(files).filter(f =>
      ["image/jpeg", "image/png", "image/webp"].includes(f.type)
    );
    if (!accepted.length) return;
    const newPreviews = accepted.map(file => ({
      file,
      url: URL.createObjectURL(file),
    }));
    setPreviews(prev => {
      const combined = [...prev, ...newPreviews].slice(0, 10);
      onFilesSelected(combined.map(p => p.file));
      return combined;
    });
  };

  // ── Dropzone (drag & drop only — click is intercepted) ────────────────
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept:   { "image/*": [".jpg", ".jpeg", ".png"] },
    maxFiles: 10,
    noClick:  true,   // we handle click ourselves
    onDrop:   (acceptedFiles) => {
      const newPreviews = acceptedFiles.map(file => ({
        file,
        url: URL.createObjectURL(file),
      }));
      setPreviews(prev => {
        const combined = [...prev, ...newPreviews].slice(0, 10);
        onFilesSelected(combined.map(p => p.file));
        return combined;
      });
    },
  });

  const removeFile = (index) => {
    const updated = previews.filter((_, i) => i !== index);
    setPreviews(updated);
    onFilesSelected(updated.map(p => p.file));
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
          user-select: none;
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
          gap: 0.6rem; margin-top: 1rem;
        }
        @media(max-width:400px) { .preview-grid { grid-template-columns: repeat(auto-fill, minmax(72px, 1fr)); } }
        .preview-img {
          width: 100%; height: 80px; object-fit: cover;
          border-radius: 8px; border: 1px solid var(--border); display: block;
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

        /* Source picker modal */
        .picker-overlay {
          position: fixed; inset: 0; z-index: 1000;
          background: #00000077; backdrop-filter: blur(4px);
          display: flex; align-items: flex-end; justify-content: center;
          padding: 1rem;
          animation: fadeIn 0.15s ease;
        }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .picker-sheet {
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: 16px;
          width: 100%; max-width: 420px;
          padding: 1.25rem;
          box-shadow: var(--shadow);
          animation: slideUp 0.2s ease;
        }
        @keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        .picker-btn {
          display: flex; align-items: center; gap: 1rem;
          width: 100%; padding: 0.9rem 1rem;
          background: var(--bg-elevated);
          border: 1px solid var(--border);
          border-radius: 12px; cursor: pointer;
          transition: all 0.15s; text-align: left;
        }
        .picker-btn:hover { border-color: var(--accent); background: var(--bg-hover); }
        .picker-icon {
          width: 44px; height: 44px; border-radius: 10px;
          display: flex; align-items: center; justify-content: center;
          font-size: 1.4rem; flex-shrink: 0;
        }
        .picker-cancel {
          width: 100%; padding: 0.7rem;
          background: transparent; border: 1px solid var(--border);
          border-radius: 10px; cursor: pointer;
          color: var(--text-muted); font-size: 0.88rem;
          transition: all 0.15s;
        }
        .picker-cancel:hover { border-color: var(--text-faint); color: var(--text-primary); }
      `}</style>

      {/* Hidden camera input */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        multiple
        style={{ display: "none" }}
        onChange={(e) => handleFiles(e.target.files)}
      />

      {/* Hidden gallery input */}
      <input
        ref={galleryInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        multiple
        style={{ display: "none" }}
        onChange={(e) => handleFiles(e.target.files)}
      />

      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? "active" : ""}`}
        onClick={() => setShowPicker(true)}
      >
        <input {...getInputProps()} />
        <div className="dropzone-icon">
          {isDragActive ? <span style={{ fontSize: "1.4rem" }}>⬇️</span> : (
            <img src="/camerawowo.png" alt="upload" style={{ width: "28px", height: "28px", objectFit: "contain" }} />
          )}
        </div>
        <p style={{ color: "var(--text-primary)", fontSize: "0.95rem", fontWeight: "500", marginBottom: "0.3rem" }}>
          {isDragActive ? "Release to upload" : "Tap to add car images"}
        </p>
        <p style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>
          or drag & drop · JPG, PNG · up to 10
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

      {/* Preview grid */}
      {previews.length > 0 && (
        <div className="preview-grid">
          {previews.map((preview, index) => (
            <div key={index} style={{ position: "relative" }}>
              <img src={preview.url} alt={`preview-${index}`} className="preview-img" />
              <button className="remove-btn"
                onClick={(e) => { e.stopPropagation(); removeFile(index); }}>
                ×
              </button>
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

      {/* Source picker bottom sheet */}
      {showPicker && (
        <div className="picker-overlay" onClick={() => setShowPicker(false)}>
          <div className="picker-sheet" onClick={(e) => e.stopPropagation()}>

            <p style={{ fontSize: "0.78rem", color: "var(--text-faint)", textAlign: "center", marginBottom: "0.85rem", letterSpacing: "0.04em", textTransform: "uppercase" }}>
              Add images
            </p>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginBottom: "0.75rem" }}>

              {/* Take photo */}
              <button className="picker-btn" onClick={() => {
                setShowPicker(false);
                setTimeout(() => cameraInputRef.current?.click(), 50);
              }}>
                <div className="picker-icon" style={{ background: "#3b82f618", border: "1px solid #3b82f633" }}>
                  📷
                </div>
                <div>
                  <p style={{ color: "var(--text-primary)", fontSize: "0.9rem", fontWeight: "600", marginBottom: "2px" }}>
                    Take a photo
                  </p>
                  <p style={{ color: "var(--text-muted)", fontSize: "0.76rem" }}>
                    Open camera and capture now
                  </p>
                </div>
              </button>

              {/* Upload from gallery */}
              <button className="picker-btn" onClick={() => {
                setShowPicker(false);
                setTimeout(() => galleryInputRef.current?.click(), 50);
              }}>
                <div className="picker-icon" style={{ background: "#a855f718", border: "1px solid #a855f733" }}>
                  🖼️
                </div>
                <div>
                  <p style={{ color: "var(--text-primary)", fontSize: "0.9rem", fontWeight: "600", marginBottom: "2px" }}>
                    Upload from gallery
                  </p>
                  <p style={{ color: "var(--text-muted)", fontSize: "0.76rem" }}>
                    Choose images from your device
                  </p>
                </div>
              </button>
            </div>

            <button className="picker-cancel" onClick={() => setShowPicker(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </>
  );
}