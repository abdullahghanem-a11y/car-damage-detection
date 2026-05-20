/**
 * ImageEditor.jsx
 *
 * Mobile-first image editor using react-easy-crop.
 * - Loads raw image from URL (served from disk, no detection labels)
 * - Pinch/drag/zoom gestures
 * - Brightness, contrast, saturation via CSS filters
 * - Rotation (quick 90° + fine slider)
 * - Save → Replace current OR Save as copy (with name + confirmation)
 *
 * Install: npm install react-easy-crop
 */

import { useState, useCallback, useEffect } from "react";
import Cropper from "react-easy-crop";

// ── Canvas export ──────────────────────────────────────────────────────────
async function getCroppedImage(imageSrc, pixelCrop, rotation, filterStr) {
  return new Promise((resolve, reject) => {
    const image  = new Image();
    image.crossOrigin = "anonymous"; // needed for URLs from same server
    image.onload = () => {
      const maxSize = Math.max(image.width, image.height);
      const safe    = 2 * ((maxSize / 2) * Math.sqrt(2));

      // Step 1: rotate + filter onto safe canvas
      const rotCanvas     = document.createElement("canvas");
      rotCanvas.width     = safe;
      rotCanvas.height    = safe;
      const rotCtx        = rotCanvas.getContext("2d");
      rotCtx.translate(safe / 2, safe / 2);
      rotCtx.rotate((rotation * Math.PI) / 180);
      rotCtx.translate(-safe / 2, -safe / 2);
      rotCtx.filter = filterStr;
      rotCtx.drawImage(image, safe / 2 - image.width / 2, safe / 2 - image.height / 2);

      // Step 2: extract crop area
      const data          = rotCtx.getImageData(0, 0, safe, safe);
      const cropCanvas    = document.createElement("canvas");
      cropCanvas.width    = pixelCrop.width;
      cropCanvas.height   = pixelCrop.height;
      const cropCtx       = cropCanvas.getContext("2d");
      cropCtx.filter      = "none";
      cropCtx.putImageData(
        data,
        Math.round(0 - safe / 2 + image.width  / 2 - pixelCrop.x),
        Math.round(0 - safe / 2 + image.height / 2 - pixelCrop.y),
      );

      cropCanvas.toBlob(
        (blob) => {
          if (!blob) { reject(new Error("toBlob failed")); return; }
          const reader      = new FileReader();
          reader.onloadend  = () => resolve(reader.result.split(",")[1]); // base64
          reader.readAsDataURL(blob);
        },
        "image/jpeg", 0.92,
      );
    };
    image.onerror = () => reject(new Error("Failed to load image for export"));
    image.src     = imageSrc;
  });
}

// ── Slider ─────────────────────────────────────────────────────────────────
function Slider({ icon, label, value, min, max, unit = "%", defaultValue, onChange, onReset }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "5px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>{icon} {label}</span>
        <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
          <span style={{ fontSize: "0.74rem", color: "var(--accent)", fontWeight: "700", minWidth: "36px", textAlign: "right" }}>
            {value}{unit}
          </span>
          {value !== defaultValue && (
            <button onClick={onReset} title="Reset"
              style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-faint)", fontSize: "0.7rem", padding: "0 2px", lineHeight: 1 }}>
              ↺
            </button>
          )}
        </div>
      </div>
      <div style={{ position: "relative", height: "26px", display: "flex", alignItems: "center" }}>
        <div style={{ position: "absolute", left: 0, right: 0, height: "4px", background: "var(--bg-elevated)", borderRadius: "2px", overflow: "hidden" }}>
          <div style={{ width: `${pct}%`, height: "100%", background: "var(--accent)", borderRadius: "2px" }} />
        </div>
        <input type="range" min={min} max={max} value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          style={{ position: "absolute", width: "100%", appearance: "none", WebkitAppearance: "none", background: "transparent", cursor: "pointer", margin: 0 }}
        />
      </div>
    </div>
  );
}

// ── Aspect ratio options ───────────────────────────────────────────────────
const ASPECTS = [
  { label: "Free", value: undefined },
  { label: "1:1",  value: 1         },
  { label: "4:3",  value: 4 / 3     },
  { label: "16:9", value: 16 / 9    },
  { label: "3:4",  value: 3 / 4     },
];

const TABS = ["Crop", "Adjust", "Rotate"];

// ── Save Mode Modal ────────────────────────────────────────────────────────
function SaveModeModal({ currentName, sessionName, onReplace, onCopy, onCancel }) {
  const [mode,    setMode]    = useState(null);
  const [copyName,setCopyName]= useState("");
  const [confirm, setConfirm] = useState(false);

  const canProceed = mode === "replace" || (mode === "copy" && copyName.trim());

  if (confirm) {
    return (
      <div style={{ position:"fixed", inset:0, zIndex:2000, background:"#00000099", backdropFilter:"blur(4px)", display:"flex", alignItems:"center", justifyContent:"center", padding:"1rem" }}>
        <div style={{ background:"var(--bg-surface)", border:"1px solid var(--border)", borderRadius:"14px", padding:"1.5rem", maxWidth:"360px", width:"100%", boxShadow:"var(--shadow)" }}>
          <p style={{ color:"var(--text-primary)", fontSize:"0.92rem", lineHeight:"1.55", marginBottom:"1.25rem" }}>
            {mode === "replace"
              ? <>Are you sure you want to <strong>replace</strong> <em>{currentName}</em> with the edited version? The original will be lost.</>
              : <>Save a copy named <strong>"{copyName}"</strong> to session <strong>{sessionName}</strong>? The original stays unchanged.</>
            }
          </p>
          <div style={{ display:"flex", gap:"0.6rem", justifyContent:"flex-end" }}>
            <button onClick={() => setConfirm(false)}
              style={{ background:"transparent", color:"var(--text-muted)", border:"1px solid var(--border)", borderRadius:"8px", padding:"0.5rem 1rem", cursor:"pointer", fontSize:"0.85rem" }}>
              ← Back
            </button>
            <button onClick={() => mode === "replace" ? onReplace() : onCopy(copyName.trim())}
              style={{ background: mode === "replace" ? "var(--danger)" : "var(--accent)", color:"white", border:"none", borderRadius:"8px", padding:"0.5rem 1.25rem", cursor:"pointer", fontSize:"0.85rem", fontWeight:"600" }}>
              {mode === "replace" ? "Replace" : "Save Copy"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ position:"fixed", inset:0, zIndex:2000, background:"#00000099", backdropFilter:"blur(4px)", display:"flex", alignItems:"center", justifyContent:"center", padding:"1rem" }}>
      <div style={{ background:"var(--bg-surface)", border:"1px solid var(--border)", borderRadius:"14px", padding:"1.5rem", maxWidth:"400px", width:"100%", boxShadow:"var(--shadow)" }}>
        <h3 style={{ color:"var(--text-primary)", fontSize:"0.95rem", fontWeight:"700", marginBottom:"0.4rem" }}>Save edited image</h3>
        <p style={{ color:"var(--text-muted)", fontSize:"0.82rem", marginBottom:"1.25rem" }}>How would you like to save your changes?</p>

        <div style={{ display:"flex", flexDirection:"column", gap:"0.6rem", marginBottom:"1.25rem" }}>
          {/* Replace option */}
          <label style={{ display:"flex", gap:"0.75rem", alignItems:"flex-start", padding:"0.85rem", background:"var(--bg-elevated)", borderRadius:"10px", border:`1px solid ${mode==="replace"?"var(--danger)":"var(--border)"}`, cursor:"pointer", transition:"border-color 0.15s" }}>
            <input type="radio" name="saveMode" value="replace" checked={mode==="replace"} onChange={() => setMode("replace")}
              style={{ marginTop:"2px", accentColor:"var(--danger)" }} />
            <div>
              <p style={{ color:"var(--text-primary)", fontSize:"0.85rem", fontWeight:"600", marginBottom:"2px" }}>Replace current</p>
              <p style={{ color:"var(--text-muted)", fontSize:"0.77rem" }}>Overwrites <em>{currentName}</em> — original will be lost</p>
            </div>
          </label>

          {/* Copy option */}
          <label style={{ display:"flex", gap:"0.75rem", alignItems:"flex-start", padding:"0.85rem", background:"var(--bg-elevated)", borderRadius:"10px", border:`1px solid ${mode==="copy"?"var(--accent)":"var(--border)"}`, cursor:"pointer", transition:"border-color 0.15s" }}>
            <input type="radio" name="saveMode" value="copy" checked={mode==="copy"} onChange={() => setMode("copy")}
              style={{ marginTop:"2px", accentColor:"var(--accent)" }} />
            <div style={{ width:"100%" }}>
              <p style={{ color:"var(--text-primary)", fontSize:"0.85rem", fontWeight:"600", marginBottom:"2px" }}>Save as copy</p>
              <p style={{ color:"var(--text-muted)", fontSize:"0.77rem", marginBottom: mode==="copy" ? "0.5rem" : 0 }}>
                Keeps original, creates new entry and re-detects
              </p>
              {mode === "copy" && (
                <input
                  autoFocus
                  placeholder='e.g. "Cropped front view"'
                  value={copyName}
                  onChange={(e) => setCopyName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && copyName.trim()) setConfirm(true); }}
                  style={{ width:"100%", background:"var(--bg-surface)", border:"1px solid var(--accent)", borderRadius:"6px", padding:"5px 8px", color:"var(--text-primary)", fontSize:"0.82rem", outline:"none", marginTop:"4px" }}
                />
              )}
            </div>
          </label>
        </div>

        <div style={{ display:"flex", gap:"0.6rem", justifyContent:"flex-end" }}>
          <button onClick={onCancel}
            style={{ background:"transparent", color:"var(--text-muted)", border:"1px solid var(--border)", borderRadius:"8px", padding:"0.5rem 1rem", cursor:"pointer", fontSize:"0.85rem" }}>
            Cancel
          </button>
          <button onClick={() => setConfirm(true)} disabled={!canProceed}
            style={{ background: canProceed ? "var(--accent)" : "var(--bg-elevated)", color: canProceed ? "white" : "var(--text-faint)", border:"none", borderRadius:"8px", padding:"0.5rem 1.25rem", cursor: canProceed ? "pointer" : "not-allowed", fontSize:"0.85rem", fontWeight:"600" }}>
            Continue →
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main ImageEditor ───────────────────────────────────────────────────────
export default function ImageEditor({ rawImageSrc, filename, sessionName, onReplace, onCopy, onCancel, saving }) {
  const [crop,          setCrop]          = useState({ x: 0, y: 0 });
  const [zoom,          setZoom]          = useState(1);
  const [rotation,      setRotation]      = useState(0);
  const [croppedArea,   setCroppedArea]   = useState(null);
  const [aspect,        setAspect]        = useState(undefined);
  const [activeTab,     setActiveTab]     = useState("Crop");
  const [brightness,    setBrightness]    = useState(100);
  const [contrast,      setContrast]      = useState(100);
  const [saturation,    setSaturation]    = useState(100);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [processing,    setProcessing]    = useState(false);
  const [imageSrc,      setImageSrc]      = useState(null);   // base64 loaded from URL
  const [imgError,      setImgError]      = useState(null);

  // Load image URL → base64 on mount to avoid canvas cross-origin taint
  useEffect(() => {
    if (!rawImageSrc) return;
    setImageSrc(null);
    setImgError(null);
    fetch(rawImageSrc)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.blob();
      })
      .then(blob => {
        const reader    = new FileReader();
        reader.onloadend = () => setImageSrc(reader.result); // full data URL
        reader.readAsDataURL(blob);
      })
      .catch(err => {
        console.error("Failed to load raw image:", err);
        setImgError("Failed to load image. Please try again.");
      });
  }, [rawImageSrc]);

  const cssFilter = `brightness(${brightness}%) contrast(${contrast}%) saturate(${saturation}%)`;

  const onCropComplete = useCallback((_, pixels) => setCroppedArea(pixels), []);

  const exportImage = async () => {
    if (!croppedArea) throw new Error("No crop area selected");
    return getCroppedImage(imageSrc, croppedArea, rotation, cssFilter);
  };

  const handleReplace = async () => {
    setShowSaveModal(false);
    setProcessing(true);
    try {
      const b64 = await exportImage();
      await onReplace(b64);
    } catch (err) {
      console.error("Replace error:", err);
      alert("Failed to save. Please try again.");
    } finally {
      setProcessing(false);
    }
  };

  const handleCopy = async (copyName) => {
    setShowSaveModal(false);
    setProcessing(true);
    try {
      const b64 = await exportImage();
      await onCopy(b64, copyName);
    } catch (err) {
      console.error("Copy error:", err);
      alert("Failed to save copy. Please try again.");
    } finally {
      setProcessing(false);
    }
  };

  const resetAll = () => {
    setCrop({ x: 0, y: 0 }); setZoom(1); setRotation(0);
    setBrightness(100); setContrast(100); setSaturation(100);
    setAspect(undefined);
  };

  return (
    <>
      <style>{`
        .ed-wrap{display:flex;flex-direction:column;height:100%;min-height:0;}
        .ed-crop{position:relative;flex:1;min-height:220px;background:#000;}
        .ed-tabs{display:flex;border-bottom:1px solid var(--border);flex-shrink:0;}
        .ed-tab{flex:1;padding:0.6rem;background:none;border:none;cursor:pointer;font-size:0.8rem;color:var(--text-muted);border-bottom:2px solid transparent;transition:all 0.15s;}
        .ed-tab.on{color:var(--accent);border-bottom-color:var(--accent);font-weight:600;}
        .ed-tab:hover:not(.on){color:var(--text-secondary);}
        .ed-panel{padding:0.85rem 1rem;display:flex;flex-direction:column;gap:0.8rem;overflow-y:auto;flex-shrink:0;max-height:230px;}
        .asp-pills{display:flex;gap:0.35rem;flex-wrap:wrap;}
        .asp-pill{background:var(--bg-elevated);border:1px solid var(--border);border-radius:6px;padding:3px 10px;cursor:pointer;font-size:0.74rem;color:var(--text-muted);transition:all 0.15s;}
        .asp-pill.on{background:var(--accent)22;border-color:var(--accent);color:var(--accent);font-weight:600;}
        .rot-row{display:flex;gap:0.5rem;}
        .rot-btn{flex:1;background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;padding:0.5rem;cursor:pointer;font-size:0.82rem;color:var(--text-muted);transition:all 0.15s;text-align:center;}
        .rot-btn:hover{border-color:var(--accent);color:var(--accent);}
        .rst-btn{background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;padding:0.45rem 0.85rem;cursor:pointer;font-size:0.76rem;color:var(--text-muted);align-self:flex-start;transition:all 0.15s;}
        .rst-btn:hover{border-color:var(--text-faint);color:var(--text-primary);}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:var(--accent);cursor:pointer;border:2px solid var(--bg-surface);box-shadow:0 1px 4px rgba(0,0,0,0.3);}
        input[type=range]::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:var(--accent);cursor:pointer;border:2px solid var(--bg-surface);}
      `}</style>

      <div className="ed-wrap">

        {/* Cropper — loads raw image from URL */}
        <div className="ed-crop">
          {imgError && (
            <div style={{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",zIndex:10,background:"var(--bg-elevated)"}}>
              <p style={{color:"var(--danger)",fontSize:"0.85rem",textAlign:"center",padding:"1rem"}}>{imgError}</p>
            </div>
          )}
          {!imageSrc && !imgError && (
            <div style={{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",zIndex:10,background:"#000"}}>
              <p style={{color:"var(--text-faint)",fontSize:"0.82rem"}}>Loading image...</p>
            </div>
          )}
          <Cropper
            image={imageSrc || rawImageSrc}
            crop={crop}
            zoom={zoom}
            rotation={rotation}
            aspect={aspect}
            onCropChange={setCrop}
            onZoomChange={setZoom}
            onRotationChange={setRotation}
            onCropComplete={onCropComplete}
            showGrid
            zoomWithScroll
            style={{
              containerStyle: { background: "#000" },
              mediaStyle:     { filter: cssFilter },
            }}
          />
        </div>

        {/* Tabs */}
        <div className="ed-tabs">
          {TABS.map(t => (
            <button key={t} className={`ed-tab ${activeTab===t?"on":""}`} onClick={() => setActiveTab(t)}>{t}</button>
          ))}
        </div>

        {/* Controls */}
        <div className="ed-panel">

          {activeTab === "Crop" && (
            <>
              <div>
                <p style={{ fontSize:"0.67rem", color:"var(--text-faint)", letterSpacing:"0.05em", textTransform:"uppercase", marginBottom:"7px" }}>Aspect Ratio</p>
                <div className="asp-pills">
                  {ASPECTS.map(a => (
                    <button key={a.label} className={`asp-pill ${aspect===a.value?"on":""}`} onClick={() => setAspect(a.value)}>
                      {a.label}
                    </button>
                  ))}
                </div>
              </div>
              <Slider icon="🔍" label="Zoom" value={Math.round(zoom*100)} min={100} max={300} defaultValue={100}
                onChange={(v) => setZoom(v/100)} onReset={() => setZoom(1)} />
              <p style={{ fontSize:"0.71rem", color:"var(--text-faint)", marginTop:"-4px" }}>
                Pinch to zoom · Drag to reposition · Resize crop corners
              </p>
            </>
          )}

          {activeTab === "Adjust" && (
            <>
              <Slider icon="☀️" label="Brightness" value={brightness} min={20}  max={200} defaultValue={100} onChange={setBrightness} onReset={() => setBrightness(100)} />
              <Slider icon="◑"  label="Contrast"   value={contrast}   min={50}  max={200} defaultValue={100} onChange={setContrast}   onReset={() => setContrast(100)}   />
              <Slider icon="🎨" label="Saturation"  value={saturation} min={0}   max={200} defaultValue={100} onChange={setSaturation} onReset={() => setSaturation(100)} />
            </>
          )}

          {activeTab === "Rotate" && (
            <>
              <div className="rot-row">
                <button className="rot-btn" onClick={() => setRotation(r => r - 90)}>↺ 90° Left</button>
                <button className="rot-btn" onClick={() => setRotation(r => r + 90)}>↻ 90° Right</button>
              </div>
              <Slider icon="↻" label="Fine Rotation" value={rotation} min={-180} max={180} unit="°" defaultValue={0}
                onChange={setRotation} onReset={() => setRotation(0)} />
            </>
          )}

          <button className="rst-btn" onClick={resetAll}>↺ Reset all</button>
        </div>

        {/* Footer */}
        <div style={{ padding:"0.75rem 1rem", borderTop:"1px solid var(--border)", display:"flex", gap:"0.6rem", justifyContent:"flex-end", flexShrink:0 }}>
          <button onClick={onCancel}
            style={{ background:"transparent", color:"var(--text-muted)", border:"1px solid var(--border)", borderRadius:"8px", padding:"0.5rem 1rem", cursor:"pointer", fontSize:"0.85rem" }}>
            Cancel
          </button>
          <button onClick={() => setShowSaveModal(true)} disabled={processing || saving}
            style={{ background:(processing||saving)?"var(--bg-elevated)":"var(--accent)", color:(processing||saving)?"var(--text-faint)":"white", border:"none", borderRadius:"8px", padding:"0.5rem 1.25rem", cursor:(processing||saving)?"not-allowed":"pointer", fontSize:"0.85rem", fontWeight:"600" }}>
            {processing ? "Processing..." : saving ? "Saving..." : "Save →"}
          </button>
        </div>
      </div>

      {showSaveModal && (
        <SaveModeModal
          currentName={filename}
          sessionName={sessionName}
          onReplace={handleReplace}
          onCopy={handleCopy}
          onCancel={() => setShowSaveModal(false)}
        />
      )}
    </>
  );
}