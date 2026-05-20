import { useState, useEffect, useRef, useCallback } from "react";
import ImageEditor from "../components/ImageEditor";
import {
  getSessions, getSession,
  renameSession   as apiRenameSession,
  deleteSession   as apiDeleteSession,
  deleteSessionImages as apiDeleteImages,
  updateDetectionImage,
  reDetectImage,
  copyDetectionImage,
} from "../services/api";

// ── Constants ──────────────────────────────────────────────────────────────
const CLASS_COLORS = {
  dent:"#ff6347", scratch:"#3b82f6", crack:"#22c55e",
  glass_shatter:"#a855f7", lamp_broken:"#eab308", tire_flat:"#f97316",
};
const CLASS_LABELS = {
  dent:"Dent", scratch:"Scratch", crack:"Crack",
  glass_shatter:"Glass Shatter", lamp_broken:"Lamp Broken", tire_flat:"Tire Flat",
};
const SEVERITY_COLORS = {
  Minor:"#22c55e", Moderate:"#f59e0b", Severe:"#ef4444",
  Rejected:"#64748b", None:"#64748b",
};
const SEVERITIES   = ["Minor","Moderate","Severe"];
const SORT_OPTIONS = [
  {value:"date_desc",      label:"Date: Newest first"},
  {value:"date_asc",       label:"Date: Oldest first"},
  {value:"name_asc",       label:"Name: A → Z"},
  {value:"name_desc",      label:"Name: Z → A"},
  {value:"severity_desc",  label:"Severity: High → Low"},
  {value:"severity_asc",   label:"Severity: Low → High"},
  {value:"instances_desc", label:"Instances: Most → Least"},
  {value:"instances_asc",  label:"Instances: Least → Most"},
];

const formatDate = (d) => new Date(d).toLocaleString();

// ── Shared styles ──────────────────────────────────────────────────────────
const GLOBAL_CSS = `
  .h-card{background:var(--bg-surface);border:1px solid var(--border);border-radius:10px;padding:0.85rem;display:flex;flex-direction:column;gap:0.5rem;transition:border-color 0.15s;}
  .h-card:hover{border-color:var(--text-faint);}
  .act-btn{background:var(--bg-elevated);border:1px solid var(--border);border-radius:6px;padding:3px 9px;cursor:pointer;font-size:0.72rem;color:var(--text-muted);transition:all 0.15s;white-space:nowrap;}
  .act-btn:hover{border-color:var(--text-faint);color:var(--text-primary);}
  .act-btn.danger:hover{border-color:var(--danger);color:var(--danger);}
  .act-btn:disabled{opacity:0.4;cursor:not-allowed;}
  .h-table{width:100%;border-collapse:collapse;}
  .h-table tr{transition:background 0.15s;}
  .h-table tr:hover td{background:var(--bg-hover)!important;}
  .pg-btn{background:transparent;color:var(--text-muted);border:1px solid var(--border);border-radius:6px;padding:0.4rem 0.85rem;cursor:pointer;font-size:0.8rem;transition:all 0.2s;}
  .pg-btn:hover:not(:disabled){border-color:var(--text-faint);color:var(--text-secondary);}
  .pg-btn:disabled{opacity:0.3;cursor:not-allowed;}
  .s-input{background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;padding:0.5rem 0.85rem 0.5rem 2.2rem;color:var(--text-primary);font-size:0.85rem;outline:none;transition:border-color 0.2s;width:100%;}
  .s-input:focus{border-color:var(--accent);}
  .s-input::placeholder{color:var(--text-faint);}
  .f-select{background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;padding:0.5rem 0.75rem;color:var(--text-primary);font-size:0.82rem;outline:none;cursor:pointer;}
  .f-select:focus{border-color:var(--accent);}
  .r-input{background:var(--bg-elevated);border:1px solid var(--accent);border-radius:6px;padding:4px 8px;color:var(--text-primary);font-size:0.82rem;outline:none;min-width:0;flex:1;}
  .overlay{position:fixed;inset:0;z-index:1000;background:#00000099;backdrop-filter:blur(5px);display:flex;align-items:center;justify-content:center;padding:1rem;}
  .modal{background:var(--bg-surface);border:1px solid var(--border);border-radius:16px;width:100%;box-shadow:var(--shadow);display:flex;flex-direction:column;overflow:hidden;}
  .modal-hd{padding:1rem 1.25rem;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-shrink:0;}
  .modal-ft{padding:0.85rem 1.25rem;border-top:1px solid var(--border);display:flex;justify-content:flex-end;gap:0.6rem;flex-shrink:0;}
  .btn-cancel{background:transparent;color:var(--text-muted);border:1px solid var(--border);border-radius:8px;padding:0.5rem 1rem;cursor:pointer;font-size:0.85rem;}
  .btn-primary{background:var(--accent);color:white;border:none;border-radius:8px;padding:0.5rem 1.25rem;cursor:pointer;font-size:0.85rem;font-weight:600;}
  .btn-danger{background:var(--danger);color:white;border:none;border-radius:8px;padding:0.5rem 1rem;cursor:pointer;font-size:0.85rem;font-weight:600;}
  .btn-primary:disabled,.btn-danger:disabled{opacity:0.4;cursor:not-allowed;}
  .table-wr{display:block;} .cards-wr{display:none;}
  @media(max-width:600px){.table-wr{display:none;}.cards-wr{display:flex;flex-direction:column;gap:0.65rem;}}
  .rename-row{display:flex;gap:0.4rem;align-items:center;min-width:0;}
`;

// ── Confirm Modal ──────────────────────────────────────────────────────────
function ConfirmModal({message, onConfirm, onCancel, confirmLabel="Confirm", danger=false, loading=false}) {
  return (
    <div className="overlay">
      <div className="modal" style={{maxWidth:"380px"}}>
        <div style={{padding:"1.5rem"}}>
          <p style={{color:"var(--text-primary)",fontSize:"0.92rem",lineHeight:"1.55",marginBottom:"1.25rem"}}>{message}</p>
          <div style={{display:"flex",gap:"0.6rem",justifyContent:"flex-end"}}>
            <button className="btn-cancel" onClick={onCancel} disabled={loading}>Cancel</button>
            <button className={danger?"btn-danger":"btn-primary"} onClick={onConfirm} disabled={loading}>
              {loading ? "..." : confirmLabel}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── View Modal ─────────────────────────────────────────────────────────────
function ViewModal({session, onClose}) {
  const images  = session.detections || [];
  const [cur, setCur] = useState(0);
  const img = images[cur];

  return (
    <div className="overlay">
      <div className="modal" style={{maxWidth:"720px",maxHeight:"92vh"}}>
        <div className="modal-hd">
          <div>
            <span style={{color:"var(--text-primary)",fontSize:"0.95rem",fontWeight:"700"}}>{session.name}</span>
            <span style={{color:"var(--text-faint)",fontSize:"0.78rem",marginLeft:"0.5rem"}}>{images.length} image{images.length!==1?"s":""}</span>
          </div>
          <button className="act-btn" onClick={onClose}>✕</button>
        </div>

        {/* Main image */}
        <div style={{background:"#000",display:"flex",alignItems:"center",justifyContent:"center",minHeight:"220px",flex:1,overflow:"hidden"}}>
          {img?.annotated_image
            ? <img src={`data:image/jpeg;base64,${img.annotated_image}`} style={{maxWidth:"100%",maxHeight:"380px",objectFit:"contain"}} />
            : <p style={{color:"var(--text-faint)",fontSize:"0.85rem"}}>No annotated image</p>
          }
        </div>

        {/* Thumbnails strip */}
        {images.length > 1 && (
          <div style={{display:"flex",gap:"0.5rem",padding:"0.65rem 1rem",overflowX:"auto",borderTop:"1px solid var(--border)",flexShrink:0}}>
            {images.map((im,i) => (
              <div key={im.id} onClick={() => setCur(i)} style={{flexShrink:0,width:"68px",height:"50px",border:`2px solid ${cur===i?"var(--accent)":"var(--border)"}`,borderRadius:"6px",overflow:"hidden",cursor:"pointer"}}>
                {im.annotated_image
                  ? <img src={`data:image/jpeg;base64,${im.annotated_image}`} style={{width:"100%",height:"100%",objectFit:"cover"}} />
                  : <div style={{width:"100%",height:"100%",background:"var(--bg-elevated)"}} />
                }
              </div>
            ))}
          </div>
        )}

        {/* Detection details — like ResultCard */}
        {img?.results?.length > 0 && (
          <div style={{padding:"0.85rem 1rem",borderTop:"1px solid var(--border)",maxHeight:"180px",overflowY:"auto",flexShrink:0}}>
            <p style={{fontSize:"0.67rem",color:"var(--text-faint)",letterSpacing:"0.05em",textTransform:"uppercase",marginBottom:"0.5rem"}}>Detections — {img.image_filename}</p>
            <div style={{display:"flex",flexDirection:"column",gap:"0.35rem"}}>
              {img.results.map((det,i) => {
                const color = CLASS_COLORS[det.class_name] || "#64748b";
                return (
                  <div key={i} style={{display:"flex",justifyContent:"space-between",alignItems:"center",background:"var(--bg-elevated)",borderRadius:"8px",padding:"0.45rem 0.65rem",borderLeft:`3px solid ${color}`,gap:"0.5rem"}}>
                    <div style={{display:"flex",alignItems:"center",gap:"0.4rem",minWidth:0}}>
                      <div style={{width:"7px",height:"7px",borderRadius:"50%",background:color,flexShrink:0}} />
                      <span style={{fontSize:"0.8rem",color:"var(--text-primary)",fontWeight:"500",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
                        {CLASS_LABELS[det.class_name]||det.class_name}
                      </span>
                    </div>
                    <div style={{display:"flex",alignItems:"center",gap:"0.5rem",flexShrink:0}}>
                      {/* Severity badge */}
                      {det.severity_label && det.severity_label!=="None" && (
                        <span style={{background:det.severity_color+"18",color:det.severity_color,border:`1px solid ${det.severity_color}44`,borderRadius:"6px",padding:"1px 7px",fontSize:"0.68rem",fontWeight:"700",textTransform:"uppercase",letterSpacing:"0.04em"}}>
                          {det.severity_label}
                        </span>
                      )}
                      {/* Confidence */}
                      <div style={{display:"flex",flexDirection:"column",alignItems:"flex-end",gap:"2px"}}>
                        <span style={{fontSize:"0.6rem",color:"var(--text-faint)",letterSpacing:"0.04em"}}>CONFIDENCE</span>
                        <div style={{display:"flex",alignItems:"center",gap:"0.35rem"}}>
                          <div style={{width:"46px",height:"4px",borderRadius:"2px",background:"var(--bg-surface)",overflow:"hidden"}}>
                            <div style={{width:`${(det.confidence*100).toFixed(0)}%`,height:"100%",background:color,borderRadius:"2px"}} />
                          </div>
                          <span style={{fontSize:"0.72rem",fontWeight:"600",color,minWidth:"28px",textAlign:"right"}}>
                            {(det.confidence*100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <div className="modal-ft">
          <button className="btn-cancel" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

// ── Image Editor Modal ─────────────────────────────────────────────────────
function ImageEditorModal({session, onClose, onSaved}) {
  // Only show images that have raw image_data stored
  const images  = session.detections?.filter(d => d.verified) || [];
  const [selIdx,  setSelIdx]  = useState(0);
  const [saving,  setSaving]  = useState(false);

  const current = images[selIdx];

  // ── Replace: overwrite current detection with edited image ─────────────
  const handleReplace = async (base64ImageData) => {
    setSaving(true);
    try {
      await updateDetectionImage(session.id, current.id, base64ImageData);
      await reDetectImage(session.id, current.id);
      onSaved();
      onClose();
    } catch (err) {
      console.error("Replace error:", err);
      alert("Failed to replace. Please try again.");
      setSaving(false);
    }
  };

  // ── Copy: create new detection entry with edited image ─────────────────
  const handleCopy = async (base64ImageData, copyName) => {
    setSaving(true);
    try {
      await copyDetectionImage(session.id, current.id, base64ImageData, copyName);
      onSaved();
      onClose();
    } catch (err) {
      console.error("Copy error:", err);
      alert("Failed to save copy. Please try again.");
      setSaving(false);
    }
  };

  if (!current) return (
    <div className="overlay">
      <div className="modal" style={{maxWidth:"380px",padding:"1.5rem"}}>
        <p style={{color:"var(--text-muted)",fontSize:"0.88rem",marginBottom:"1rem"}}>
          No editable images found. Raw image data must be stored during detection.
        </p>
        <button className="btn-cancel" onClick={onClose}>Close</button>
      </div>
    </div>
  );

  return (
    <div className="overlay">
      <div className="modal" style={{maxWidth:"520px",maxHeight:"96vh"}}>
        <div className="modal-hd" style={{flexShrink:0}}>
          <div>
            <span style={{color:"var(--text-primary)",fontSize:"0.95rem",fontWeight:"700"}}>Edit — {session.name}</span>
            {images.length > 1 && (
              <span style={{color:"var(--text-faint)",fontSize:"0.78rem",marginLeft:"0.5rem"}}>{selIdx+1}/{images.length}</span>
            )}
          </div>
          <button className="act-btn" onClick={onClose}>✕</button>
        </div>

        {/* Image selector (if multiple) */}
        {images.length > 1 && (
          <div style={{display:"flex",gap:"0.4rem",padding:"0.6rem 1rem",borderBottom:"1px solid var(--border)",overflowX:"auto",flexShrink:0}}>
            {images.map((img,i) => (
              <div key={img.id} onClick={() => setSelIdx(i)}
                style={{flexShrink:0,width:"56px",height:"42px",border:`2px solid ${selIdx===i?"var(--accent)":"var(--border)"}`,borderRadius:"6px",overflow:"hidden",cursor:"pointer",transition:"border-color 0.15s"}}>
                {/* Show annotated thumbnail in selector, raw in editor */}
                <img src={`data:image/jpeg;base64,${img.annotated_image}`} style={{width:"100%",height:"100%",objectFit:"cover",display:"block"}} />
              </div>
            ))}
          </div>
        )}

        {/* Editor — passes RAW image (no detection labels) */}
        <div style={{flex:1,minHeight:0,display:"flex",flexDirection:"column"}}>
          <ImageEditor
            rawImageSrc={`${import.meta.env.VITE_API_URL || "http://localhost:8000"}/api/sessions/${session.id}/images/${current.id}/raw`}
            filename={current.image_filename}
            sessionName={session.name}
            onReplace={handleReplace}
            onCopy={handleCopy}
            onCancel={onClose}
            saving={saving}
          />
        </div>
      </div>
    </div>
  );
}

// ── Delete Modal ───────────────────────────────────────────────────────────
function DeleteModal({session, onConfirm, onCancel}) {
  const images  = session.detections?.filter(d => d.verified) || [];
  const isMulti = images.length > 1;
  const [selected, setSelected] = useState(new Set());
  const [step,     setStep]     = useState(isMulti ? "select" : "confirm");
  const [loading,  setLoading]  = useState(false);

  const toggleImg = (id) => {
    const s = new Set(selected);
    s.has(id) ? s.delete(id) : s.add(id);
    setSelected(s);
  };

  const handleConfirm = async () => {
    setLoading(true);
    try {
      await onConfirm(isMulti && selected.size < images.length ? [...selected] : null);
    } finally { setLoading(false); }
  };

  return (
    <div className="overlay">
      <div className="modal" style={{maxWidth:"420px"}}>
        <div style={{padding:"1.5rem"}}>
          {step === "select" ? (
            <>
              <h3 style={{color:"var(--text-primary)",fontSize:"0.92rem",fontWeight:"700",marginBottom:"0.5rem"}}>Delete Images</h3>
              <p style={{color:"var(--text-muted)",fontSize:"0.82rem",marginBottom:"1rem"}}>
                Select images to delete from <strong style={{color:"var(--text-primary)"}}>{session.name}</strong>. Deleting all removes the session.
              </p>
              <div style={{display:"flex",flexDirection:"column",gap:"0.4rem",marginBottom:"1.25rem",maxHeight:"220px",overflowY:"auto"}}>
                {images.map(img => (
                  <label key={img.id} style={{display:"flex",alignItems:"center",gap:"0.6rem",background:"var(--bg-elevated)",borderRadius:"8px",padding:"0.5rem 0.75rem",cursor:"pointer",border:`1px solid ${selected.has(img.id)?"var(--danger)":"var(--border)"}`}}>
                    <input type="checkbox" checked={selected.has(img.id)} onChange={() => toggleImg(img.id)} style={{accentColor:"var(--danger)",width:"15px",height:"15px"}} />
                    {img.annotated_image
                      ? <img src={`data:image/jpeg;base64,${img.annotated_image}`} style={{width:"40px",height:"30px",objectFit:"cover",borderRadius:"4px"}} />
                      : <div style={{width:"40px",height:"30px",background:"var(--bg-surface)",borderRadius:"4px"}} />
                    }
                    <span style={{fontSize:"0.8rem",color:"var(--text-secondary)",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{img.image_filename}</span>
                  </label>
                ))}
              </div>
              <div style={{display:"flex",gap:"0.6rem",justifyContent:"flex-end"}}>
                <button className="btn-cancel" onClick={onCancel}>Cancel</button>
                <button disabled={selected.size===0} onClick={() => setStep("confirm")}
                  style={{background:selected.size===0?"var(--bg-elevated)":"var(--danger)",color:selected.size===0?"var(--text-faint)":"white",border:"none",borderRadius:"8px",padding:"0.5rem 1rem",cursor:selected.size===0?"not-allowed":"pointer",fontSize:"0.85rem",fontWeight:"600"}}>
                  Next ({selected.size} selected)
                </button>
              </div>
            </>
          ) : (
            <>
              <p style={{color:"var(--text-primary)",fontSize:"0.92rem",lineHeight:"1.55",marginBottom:"1.25rem"}}>
                Are you sure you want to delete{" "}
                {isMulti && selected.size < images.length
                  ? <><strong>{selected.size} image{selected.size!==1?"s":""}</strong> from <strong>{session.name}</strong>?</>
                  : <>the session <strong>{session.name}</strong> and all its images?</>
                }{" "}This cannot be undone.
              </p>
              <div style={{display:"flex",gap:"0.6rem",justifyContent:"flex-end"}}>
                <button className="btn-cancel" onClick={() => isMulti?setStep("select"):onCancel()} disabled={loading}>
                  {isMulti?"← Back":"Cancel"}
                </button>
                <button className="btn-danger" onClick={handleConfirm} disabled={loading}>
                  {loading?"Deleting...":"Delete"}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Severity / confidence badge components ─────────────────────────────────
function SevBadge({label, color}) {
  if (!label||label==="None") return null;
  return <span style={{background:color+"18",color,border:`1px solid ${color}44`,borderRadius:"6px",padding:"1px 7px",fontSize:"0.68rem",fontWeight:"700",textTransform:"uppercase",letterSpacing:"0.04em"}}>{label}</span>;
}

// ── Main History Page ──────────────────────────────────────────────────────
export default function History() {
  const [sessions,    setSessions]    = useState([]);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState(null);
  const [skip,        setSkip]        = useState(0);

  const [search,      setSearch]      = useState("");
  const [filterSev,   setFilterSev]   = useState("");
  const [sortBy,      setSortBy]      = useState("date_desc");
  const [showFilters, setShowFilters] = useState(false);

  const [viewSess,    setViewSess]    = useState(null);
  const [editSess,    setEditSess]    = useState(null);
  const [deleteSess,  setDeleteSess]  = useState(null);
  const [renameSess,  setRenameSess]  = useState(null);
  const [renameVal,   setRenameVal]   = useState("");
  const [confirmRen,  setConfirmRen]  = useState(false);
  const [renLoading,  setRenLoading]  = useState(false);

  const LIMIT = 20;

  const fetchSessions = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const data = await getSessions({skip, limit:LIMIT, search, severity:filterSev, sort:sortBy});
      setSessions(Array.isArray(data) ? data : []);
    } catch {
      setError("Failed to load history. Is the backend running?");
    } finally { setLoading(false); }
  }, [skip, search, filterSev, sortBy]);

  useEffect(() => { fetchSessions(); }, [fetchSessions]);

  const openFull = async (s, setter) => {
    try { setter(await getSession(s.id)); }
    catch { setter(s); }
  };

  // ── Rename ─────────────────────────────────────────────────────────────
  const handleRenameConfirm = async () => {
    setRenLoading(true);
    try {
      const updated = await apiRenameSession(renameSess.id, renameVal);
      setSessions(prev => prev.map(s => s.id===renameSess.id ? {...s, name:updated.name} : s));
      setRenameSess(null); setConfirmRen(false); setRenameVal("");
    } catch { alert("Failed to rename session."); }
    finally { setRenLoading(false); }
  };

  // ── Delete ─────────────────────────────────────────────────────────────
  const handleDelete = async (detectionIds) => {
    const sessId = deleteSess.id;
    try {
      if (detectionIds === null) {
        await apiDeleteSession(sessId);
        setSessions(prev => prev.filter(s => s.id !== sessId));
      } else {
        const result = await apiDeleteImages(sessId, detectionIds);
        if (result.session_deleted) {
          setSessions(prev => prev.filter(s => s.id !== sessId));
        } else {
          fetchSessions();
        }
      }
      setDeleteSess(null);
    } catch (err) {
      console.error("Delete error:", err?.response?.data || err);
      alert(`Delete failed: ${err?.response?.data?.detail || err.message}`);
    }
  };

  const activeFilters = [filterSev].filter(Boolean).length;

  // Inline rename component (works on both desktop and mobile)
  const RenameInline = ({s}) => {
    const isRenaming = renameSess?.id === s.id;
    if (!isRenaming) return (
      <span style={{color:"var(--text-secondary)",fontSize:"0.82rem",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",display:"block"}}>{s.name}</span>
    );
    return (
      <div className="rename-row">
        <input className="r-input" value={renameVal} autoFocus
          onChange={(e) => setRenameVal(e.target.value)}
          onKeyDown={(e) => { if(e.key==="Enter"&&renameVal.trim()) setConfirmRen(true); if(e.key==="Escape"){setRenameSess(null);setRenameVal("");} }} />
        <button className="act-btn" onClick={() => renameVal.trim()&&setConfirmRen(true)} style={{color:"var(--accent)",borderColor:"var(--accent)",padding:"3px 8px"}}>✓</button>
        <button className="act-btn" onClick={() => {setRenameSess(null);setRenameVal("");}}>✕</button>
      </div>
    );
  };

  const Actions = ({s}) => (
    <div style={{display:"flex",gap:"0.3rem",flexWrap:"wrap"}}>
      <button className="act-btn" onClick={() => openFull(s, setViewSess)}>👁 View</button>
      <button className="act-btn" onClick={() => openFull(s, setEditSess)}>✏ Edit</button>
      <button className="act-btn" onClick={() => {setRenameSess(s);setRenameVal(s.name);}}>✎ Rename</button>
      <button className="act-btn danger" onClick={() => openFull(s, setDeleteSess)}>🗑 Delete</button>
    </div>
  );

  const SevPill = ({sev}) => {
    if (!sev) return <span style={{color:"var(--text-faint)"}}>—</span>;
    const c = SEVERITY_COLORS[sev]||"#64748b";
    return <span style={{background:c+"18",color:c,border:`1px solid ${c}44`,borderRadius:"6px",padding:"1px 7px",fontSize:"0.7rem",fontWeight:"700",textTransform:"uppercase",letterSpacing:"0.04em"}}>{sev}</span>;
  };

  return (
    <>
      <style>{GLOBAL_CSS}</style>
      <div style={{maxWidth:"1100px",margin:"0 auto",padding:"1rem"}}>

        <div style={{marginBottom:"1.25rem"}}>
          <h1 style={{fontSize:"1.4rem",fontWeight:"700",color:"var(--text-primary)"}}>Detection History</h1>
          <p style={{color:"var(--text-muted)",marginTop:"0.3rem",fontSize:"0.85rem"}}>All past car damage detection sessions</p>
        </div>

        {/* Search + Filter + Sort */}
        <div style={{display:"flex",flexDirection:"column",gap:"0.6rem",marginBottom:"1.25rem"}}>
          <div style={{display:"flex",gap:"0.6rem",alignItems:"center",flexWrap:"wrap"}}>
            <div style={{position:"relative",flex:1,minWidth:"160px"}}>
              <span style={{position:"absolute",left:"0.7rem",top:"50%",transform:"translateY(-50%)",color:"var(--text-faint)",fontSize:"0.85rem",pointerEvents:"none"}}>🔍</span>
              <input className="s-input" placeholder="Search sessions..." value={search}
                onChange={(e)=>{setSearch(e.target.value);setSkip(0);}} />
            </div>
            <button onClick={()=>setShowFilters(!showFilters)} className="act-btn"
              style={{padding:"0.5rem 0.85rem",fontSize:"0.82rem",borderColor:activeFilters>0?"var(--accent)":"var(--border)",color:activeFilters>0?"var(--accent)":"var(--text-muted)"}}>
              ⚙ Filters{activeFilters>0?` (${activeFilters})`:""}
            </button>
            <select className="f-select" value={sortBy} onChange={(e)=>{setSortBy(e.target.value);setSkip(0);}}>
              {SORT_OPTIONS.map(o=><option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </div>

          {showFilters && (
            <div style={{display:"flex",gap:"0.6rem",flexWrap:"wrap",padding:"0.75rem",background:"var(--bg-elevated)",borderRadius:"10px",border:"1px solid var(--border)"}}>
              <div>
                <p style={{fontSize:"0.68rem",color:"var(--text-faint)",letterSpacing:"0.05em",textTransform:"uppercase",marginBottom:"4px"}}>Severity</p>
                <select className="f-select" value={filterSev} onChange={(e)=>{setFilterSev(e.target.value);setSkip(0);}}>
                  <option value="">All severities</option>
                  {SEVERITIES.map(s=><option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              {filterSev&&<div style={{display:"flex",alignItems:"flex-end"}}>
                <button className="act-btn" onClick={()=>{setFilterSev("");setSkip(0);}} style={{padding:"0.5rem 0.85rem",fontSize:"0.82rem"}}>Clear filters</button>
              </div>}
            </div>
          )}
        </div>

        {!loading&&sessions.length>0&&(
          <p style={{fontSize:"0.75rem",color:"var(--text-faint)",marginBottom:"0.75rem"}}>
            {sessions.length} session{sessions.length!==1?"s":""}{(search||filterSev)?" matching filters":""}
          </p>
        )}

        {loading&&<div style={{textAlign:"center",color:"var(--text-muted)",padding:"2.5rem"}}><p style={{fontSize:"0.88rem"}}>Loading history...</p></div>}
        {error&&<div style={{background:"var(--bg-elevated)",border:"1px solid var(--danger)",borderRadius:"8px",padding:"0.75rem",color:"var(--danger)",fontSize:"0.82rem"}}>{error}</div>}

        {!loading&&!error&&sessions.length===0&&(
          <div style={{textAlign:"center",color:"var(--text-faint)",padding:"2.5rem"}}>
            <p style={{fontSize:"2rem",marginBottom:"0.6rem"}}>🚗</p>
            <p style={{fontSize:"0.88rem"}}>{!search&&!filterSev?"No detections yet.":"No sessions match your filters."}</p>
          </div>
        )}

        {!loading&&sessions.length>0&&(
          <>
            {/* ── Desktop Table ── */}
            <div className="table-wr" style={{background:"var(--bg-surface)",borderRadius:"12px",border:"1px solid var(--border)",overflow:"hidden",boxShadow:"var(--shadow)"}}>
              <table className="h-table">
                <thead>
                  <tr style={{background:"var(--bg-elevated)"}}>
                    {["ID","Session","Images","Severity","Date","Actions"].map(h=>(
                      <th key={h} style={{padding:"0.75rem 0.85rem",textAlign:"left",fontSize:"0.68rem",color:"var(--text-faint)",fontWeight:"700",letterSpacing:"0.06em",textTransform:"uppercase",borderBottom:"1px solid var(--border)"}}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sessions.map(s=>(
                    <tr key={s.id}>
                      <td style={{padding:"0.75rem 0.85rem",color:"var(--text-faint)",fontSize:"0.78rem",background:"var(--bg-surface)"}}>#{s.id}</td>
                      <td style={{padding:"0.75rem 0.85rem",background:"var(--bg-surface)",maxWidth:"200px"}}>
                        <RenameInline s={s} />
                      </td>
                      <td style={{padding:"0.75rem 0.85rem",background:"var(--bg-surface)"}}>
                        <span style={{background:"var(--accent)22",color:"var(--accent)",border:"1px solid var(--accent)44",borderRadius:"6px",padding:"1px 7px",fontSize:"0.73rem",fontWeight:"600"}}>{s.image_count}</span>
                      </td>
                      <td style={{padding:"0.75rem 0.85rem",background:"var(--bg-surface)"}}><SevPill sev={s.overall_severity} /></td>
                      <td style={{padding:"0.75rem 0.85rem",color:"var(--text-faint)",fontSize:"0.75rem",background:"var(--bg-surface)",whiteSpace:"nowrap"}}>{formatDate(s.created_at)}</td>
                      <td style={{padding:"0.75rem 0.85rem",background:"var(--bg-surface)"}}><Actions s={s} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"0.75rem 1rem",borderTop:"1px solid var(--border)"}}>
                <span style={{color:"var(--text-faint)",fontSize:"0.75rem"}}>{skip+1}–{skip+sessions.length}</span>
                <div style={{display:"flex",gap:"0.5rem"}}>
                  <button className="pg-btn" onClick={()=>setSkip(Math.max(0,skip-LIMIT))} disabled={skip===0}>← Prev</button>
                  <button className="pg-btn" onClick={()=>setSkip(skip+LIMIT)} disabled={sessions.length<LIMIT}>Next →</button>
                </div>
              </div>
            </div>

            {/* ── Mobile Cards ── */}
            <div className="cards-wr">
              {sessions.map(s=>{
                const c=SEVERITY_COLORS[s.overall_severity]||"#64748b";
                return (
                  <div key={s.id} className="h-card">
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:"0.5rem"}}>
                      <span style={{color:"var(--text-faint)",fontSize:"0.75rem",flexShrink:0}}>#{s.id}</span>
                      {s.overall_severity&&<span style={{background:c+"18",color:c,border:`1px solid ${c}44`,borderRadius:"6px",padding:"1px 8px",fontSize:"0.68rem",fontWeight:"700",textTransform:"uppercase",flexShrink:0}}>{s.overall_severity}</span>}
                    </div>
                    {/* Rename inline on mobile */}
                    <RenameInline s={s} />
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",flexWrap:"wrap",gap:"0.4rem"}}>
                      <span style={{background:"var(--accent)22",color:"var(--accent)",border:"1px solid var(--accent)44",borderRadius:"6px",padding:"1px 7px",fontSize:"0.72rem",fontWeight:"600"}}>{s.image_count} image{s.image_count!==1?"s":""}</span>
                      <span style={{color:"var(--text-faint)",fontSize:"0.72rem"}}>{formatDate(s.created_at)}</span>
                    </div>
                    <div style={{paddingTop:"0.25rem",borderTop:"1px solid var(--border)"}}>
                      <Actions s={s} />
                    </div>
                  </div>
                );
              })}
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"0.25rem 0"}}>
                <span style={{color:"var(--text-faint)",fontSize:"0.75rem"}}>{skip+1}–{skip+sessions.length}</span>
                <div style={{display:"flex",gap:"0.5rem"}}>
                  <button className="pg-btn" onClick={()=>setSkip(Math.max(0,skip-LIMIT))} disabled={skip===0}>← Prev</button>
                  <button className="pg-btn" onClick={()=>setSkip(skip+LIMIT)} disabled={sessions.length<LIMIT}>Next →</button>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* ── Modals ── */}
      {viewSess   && <ViewModal        session={viewSess}   onClose={()=>setViewSess(null)} />}
      {editSess   && <ImageEditorModal session={editSess}   onClose={()=>setEditSess(null)} onSaved={fetchSessions} />}
      {deleteSess && <DeleteModal      session={deleteSess} onConfirm={handleDelete} onCancel={()=>setDeleteSess(null)} />}

      {confirmRen && (
        <ConfirmModal
          message={`Are you sure you want to rename this session to "${renameVal}"?`}
          confirmLabel="Rename"
          loading={renLoading}
          onConfirm={handleRenameConfirm}
          onCancel={()=>setConfirmRen(false)}
        />
      )}
    </>
  );
}