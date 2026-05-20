import { useState, useEffect, useCallback } from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import {
  getAdminStats, getAdminHealth, getAdminUsers,
  toggleUser, getAdminLogs, getModelInfo,
  switchModel, getAdminFeedback,
} from "../services/api";

// ── Colors ─────────────────────────────────────────────────────────────────
const CLASS_COLORS = {
  dent:"#ff6347", scratch:"#3b82f6", crack:"#22c55e",
  glass_shatter:"#a855f7", lamp_broken:"#eab308", tire_flat:"#f97316",
};
const SEV_COLORS = { Minor:"#22c55e", Moderate:"#f59e0b", Severe:"#ef4444", Unknown:"#64748b" };
const TABS = ["Overview","Model","Users","Logs","Feedback"];

// ── Mini bar chart ──────────────────────────────────────────────────────────
function BarChart({ data, colorKey, labelKey, valueKey, maxVal }) {
  if (!data?.length) return <p style={{color:"var(--text-faint)",fontSize:"0.82rem"}}>No data</p>;
  const max = maxVal || Math.max(...data.map(d => d[valueKey]));
  return (
    <div style={{display:"flex",flexDirection:"column",gap:"0.4rem"}}>
      {data.map((d,i) => {
        const pct   = max > 0 ? (d[valueKey] / max) * 100 : 0;
        const color = colorKey ? (d[colorKey] || "#64748b") : Object.values(CLASS_COLORS)[i % 6];
        return (
          <div key={i} style={{display:"flex",alignItems:"center",gap:"0.6rem"}}>
            <span style={{fontSize:"0.75rem",color:"var(--text-muted)",minWidth:"90px",textAlign:"right",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
              {d[labelKey]}
            </span>
            <div style={{flex:1,height:"18px",background:"var(--bg-elevated)",borderRadius:"4px",overflow:"hidden"}}>
              <div style={{width:`${pct}%`,height:"100%",background:color,borderRadius:"4px",transition:"width 0.5s ease"}} />
            </div>
            <span style={{fontSize:"0.75rem",color:"var(--text-primary)",fontWeight:"600",minWidth:"30px"}}>{d[valueKey]}</span>
          </div>
        );
      })}
    </div>
  );
}

// ── Line chart (training curve) ─────────────────────────────────────────────
function LineChart({ data, lines, width=340, height=160 }) {
  if (!data?.length) return <p style={{color:"var(--text-faint)",fontSize:"0.82rem"}}>No data</p>;
  const pad = { t:10, r:10, b:30, l:40 };
  const W   = width  - pad.l - pad.r;
  const H   = height - pad.t - pad.b;
  const epochs = data.map(d => d.epoch);
  const minX = Math.min(...epochs), maxX = Math.max(...epochs);
  const allVals = lines.flatMap(l => data.map(d => d[l.key]));
  const minY = Math.min(...allVals) * 0.95, maxY = Math.max(...allVals) * 1.02;

  const px = (e) => pad.l + ((e - minX) / (maxX - minX || 1)) * W;
  const py = (v) => pad.t + H - ((v - minY) / (maxY - minY || 1)) * H;

  const pathFor = (key) => data.map((d,i) =>
    `${i===0?"M":"L"}${px(d.epoch).toFixed(1)},${py(d[key]).toFixed(1)}`
  ).join(" ");

  // Y axis labels
  const yTicks = [minY, (minY+maxY)/2, maxY].map(v => ({v, y: py(v)}));
  // X axis labels
  const step   = Math.ceil(data.length / 5);
  const xTicks = data.filter((_,i) => i % step === 0 || i === data.length-1);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{width:"100%",height:"auto",overflow:"visible"}}>
      {/* Grid */}
      {yTicks.map((t,i) => (
        <g key={i}>
          <line x1={pad.l} y1={t.y} x2={pad.l+W} y2={t.y} stroke="var(--border)" strokeWidth="1" strokeDasharray="3,3" />
          <text x={pad.l-4} y={t.y+4} textAnchor="end" fontSize="9" fill="var(--text-faint)">{t.v.toFixed(0)}%</text>
        </g>
      ))}
      {xTicks.map((d,i) => (
        <text key={i} x={px(d.epoch)} y={height-4} textAnchor="middle" fontSize="9" fill="var(--text-faint)">{d.epoch}</text>
      ))}

      {/* Lines */}
      {lines.map(l => (
        <path key={l.key} d={pathFor(l.key)} fill="none" stroke={l.color} strokeWidth="2" strokeLinejoin="round" />
      ))}

      {/* Legend */}
      {lines.map((l,i) => (
        <g key={l.key} transform={`translate(${pad.l + i * 90}, ${height - 8})`}>
          <rect x={0} y={-8} width={8} height={8} fill={l.color} rx={2} />
          <text x={12} y={-1} fontSize="9" fill="var(--text-muted)">{l.label}</text>
        </g>
      ))}
    </svg>
  );
}

// ── Stat card ───────────────────────────────────────────────────────────────
function StatCard({ label, value, icon, color="#3b82f6", sub }) {
  return (
    <div style={{background:"var(--bg-surface)",border:"1px solid var(--border)",borderRadius:"12px",padding:"1rem 1.25rem",display:"flex",gap:"0.85rem",alignItems:"flex-start"}}>
      <div style={{width:"40px",height:"40px",borderRadius:"10px",background:color+"22",border:`1px solid ${color}33`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:"1.2rem",flexShrink:0}}>
        {icon}
      </div>
      <div>
        <p style={{fontSize:"0.72rem",color:"var(--text-faint)",letterSpacing:"0.05em",textTransform:"uppercase",marginBottom:"2px"}}>{label}</p>
        <p style={{fontSize:"1.5rem",fontWeight:"700",color:"var(--text-primary)",lineHeight:1}}>{value?.toLocaleString() ?? "—"}</p>
        {sub && <p style={{fontSize:"0.72rem",color:"var(--text-muted)",marginTop:"3px"}}>{sub}</p>}
      </div>
    </div>
  );
}

// ── Health indicator ────────────────────────────────────────────────────────
function HealthDot({ status }) {
  const ok    = status === "connected" || status === "loaded" || status === "healthy";
  const color = ok ? "var(--success)" : "var(--danger)";
  return (
    <span style={{display:"inline-flex",alignItems:"center",gap:"5px"}}>
      <span style={{width:"8px",height:"8px",borderRadius:"50%",background:color,display:"inline-block"}} />
      <span style={{fontSize:"0.8rem",color,fontWeight:"600",textTransform:"capitalize"}}>{status}</span>
    </span>
  );
}

// ── Metric badge ────────────────────────────────────────────────────────────
function MetricBadge({ label, value, color="#3b82f6" }) {
  return (
    <div style={{background:"var(--bg-elevated)",border:`1px solid ${color}33`,borderRadius:"10px",padding:"0.75rem 1rem",textAlign:"center"}}>
      <p style={{fontSize:"0.68rem",color:"var(--text-faint)",letterSpacing:"0.05em",textTransform:"uppercase",marginBottom:"4px"}}>{label}</p>
      <p style={{fontSize:"1.4rem",fontWeight:"700",color}}>{value}%</p>
    </div>
  );
}

// ── Main Dashboard ──────────────────────────────────────────────────────────
export default function AdminDashboard() {
  const { isAdmin, loading: authLoading } = useAuth();
  const [tab,       setTab]       = useState("Overview");
  const [stats,     setStats]     = useState(null);
  const [health,    setHealth]    = useState(null);
  const [users,     setUsers]     = useState(null);
  const [logs,      setLogs]      = useState(null);
  const [model,     setModel]     = useState(null);
  const [feedback,  setFeedback]  = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);
  const [userSearch, setUserSearch] = useState("");
  const [logFilter,  setLogFilter]  = useState("");
  const [switching,  setSwitching]  = useState(false);

  if (!authLoading && !isAdmin) return <Navigate to="/" replace />;

  const load = useCallback(async (t) => {
    setLoading(true); setError(null);
    try {
      switch(t) {
        case "Overview": {
          const [s, h] = await Promise.all([getAdminStats(), getAdminHealth()]);
          setStats(s); setHealth(h); break;
        }
        case "Model":    setModel(await getModelInfo()); break;
        case "Users":    setUsers(await getAdminUsers({ search: userSearch })); break;
        case "Logs":     setLogs(await getAdminLogs({ action: logFilter })); break;
        case "Feedback": setFeedback(await getAdminFeedback()); break;
      }
    } catch(e) {
      setError(e?.response?.data?.detail || "Failed to load data");
    } finally {
      setLoading(false); }
  }, [userSearch, logFilter]);

  useEffect(() => { load(tab); }, [tab]);

  const handleTabChange = (t) => { setTab(t); };

  const handleToggleUser = async (userId, current) => {
    await toggleUser(userId, !current);
    load("Users");
  };

  const handleSwitchModel = async (version) => {
    setSwitching(true);
    try {
      await switchModel(version);
      await load("Model");
    } catch(e) {
      alert(e?.response?.data?.detail || "Failed to switch model");
    } finally { setSwitching(false); }
  };

  return (
    <>
      <style>{`
        .adm-wrap{max-width:1100px;margin:0 auto;padding:1rem;}
        .adm-tabs{display:flex;gap:0.3rem;margin-bottom:1.5rem;overflow-x:auto;padding-bottom:2px;}
        .adm-tab{padding:0.5rem 1rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;cursor:pointer;font-size:0.82rem;color:var(--text-muted);transition:all 0.15s;white-space:nowrap;}
        .adm-tab:hover{border-color:var(--text-faint);color:var(--text-primary);}
        .adm-tab.on{background:var(--accent)22;border-color:var(--accent);color:var(--accent);font-weight:600;}
        .adm-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:0.85rem;margin-bottom:1.5rem;}
        .adm-card{background:var(--bg-surface);border:1px solid var(--border);border-radius:12px;padding:1.25rem;}
        .adm-card h3{font-size:0.78rem;color:var(--text-faint);letter-spacing:0.05em;text-transform:uppercase;margin-bottom:1rem;}
        .adm-table{width:100%;border-collapse:collapse;}
        .adm-table th{padding:0.65rem 0.85rem;text-align:left;font-size:0.68rem;color:var(--text-faint);font-weight:700;letter-spacing:0.06em;text-transform:uppercase;border-bottom:1px solid var(--border);}
        .adm-table td{padding:0.65rem 0.85rem;font-size:0.8rem;color:var(--text-secondary);border-bottom:1px solid var(--border-subtle);}
        .adm-table tr:hover td{background:var(--bg-hover);}
        .adm-input{background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;padding:0.45rem 0.75rem;color:var(--text-primary);font-size:0.82rem;outline:none;transition:border-color 0.2s;}
        .adm-input:focus{border-color:var(--accent);}
        .toggle-btn{padding:3px 10px;border-radius:6px;font-size:0.72rem;font-weight:600;cursor:pointer;border:1px solid;transition:all 0.15s;}
        @media(max-width:600px){
          .adm-grid{grid-template-columns:1fr 1fr;}
          .adm-table thead{display:none;}
          .adm-table td{display:block;padding:0.4rem 0.6rem;}
        }
      `}</style>

      <div className="adm-wrap">
        <div style={{marginBottom:"1.25rem"}}>
          <h1 style={{fontSize:"1.4rem",fontWeight:"700",color:"var(--text-primary)"}}>Admin Dashboard</h1>
          <p style={{color:"var(--text-muted)",fontSize:"0.85rem",marginTop:"0.3rem"}}>System management and analytics</p>
        </div>

        {/* Tabs */}
        <div className="adm-tabs">
          {TABS.map(t => (
            <button key={t} className={`adm-tab ${tab===t?"on":""}`} onClick={() => handleTabChange(t)}>{t}</button>
          ))}
        </div>

        {error && <div style={{background:"var(--bg-elevated)",border:"1px solid var(--danger)",borderRadius:"8px",padding:"0.75rem",color:"var(--danger)",fontSize:"0.82rem",marginBottom:"1rem"}}>{error}</div>}
        {loading && <div style={{textAlign:"center",color:"var(--text-muted)",padding:"2rem",fontSize:"0.88rem"}}>Loading...</div>}

        {/* ── Overview tab ── */}
        {!loading && tab==="Overview" && stats && health && (
          <>
            {/* Stat cards */}
            <div className="adm-grid">
              <StatCard label="Total Users"      value={stats.overview.total_users}      icon="👥" color="#3b82f6" />
              <StatCard label="Total Sessions"   value={stats.overview.total_sessions}   icon="📁" color="#a855f7" />
              <StatCard label="Total Detections" value={stats.overview.total_detections} icon="🔍" color="#f59e0b" />
              <StatCard label="Damage Instances" value={stats.overview.total_instances}  icon="⚠️" color="#ef4444" />
            </div>

            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0.85rem",marginBottom:"1rem"}}>

              {/* Damage distribution */}
              <div className="adm-card">
                <h3>Damage Distribution</h3>
                <BarChart
                  data={stats.damage_distribution}
                  labelKey="class_name" valueKey="count"
                  colorKey={null}
                />
              </div>

              {/* Severity distribution */}
              <div className="adm-card">
                <h3>Severity Distribution</h3>
                <BarChart
                  data={stats.severity_distribution.map(s => ({...s, color: SEV_COLORS[s.severity]||"#64748b"}))}
                  labelKey="severity" valueKey="count" colorKey="color"
                />
              </div>
            </div>

            {/* System health */}
            <div className="adm-card" style={{marginBottom:"1rem"}}>
              <h3>System Health</h3>
              <div style={{display:"flex",flexWrap:"wrap",gap:"1.5rem"}}>
                <div>
                  <p style={{fontSize:"0.72rem",color:"var(--text-faint)",marginBottom:"4px"}}>DATABASE</p>
                  <HealthDot status={health.database.status} />
                </div>
                <div>
                  <p style={{fontSize:"0.72rem",color:"var(--text-faint)",marginBottom:"4px"}}>MODEL</p>
                  <HealthDot status={health.model.status} />
                </div>
                <div>
                  <p style={{fontSize:"0.72rem",color:"var(--text-faint)",marginBottom:"4px"}}>UPTIME</p>
                  <span style={{fontSize:"0.8rem",color:"var(--text-primary)",fontWeight:"600"}}>{health.system.uptime}</span>
                </div>
                <div>
                  <p style={{fontSize:"0.72rem",color:"var(--text-faint)",marginBottom:"4px"}}>MEMORY</p>
                  <span style={{fontSize:"0.8rem",color:"var(--text-primary)",fontWeight:"600"}}>{health.system.memory_mb} MB</span>
                </div>
                <div>
                  <p style={{fontSize:"0.72rem",color:"var(--text-faint)",marginBottom:"4px"}}>CPU</p>
                  <span style={{fontSize:"0.8rem",color:"var(--text-primary)",fontWeight:"600"}}>{health.system.cpu_pct}%</span>
                </div>
                <div>
                  <p style={{fontSize:"0.72rem",color:"var(--text-faint)",marginBottom:"4px"}}>ACTIVE MODEL</p>
                  <span style={{fontSize:"0.8rem",color:"var(--accent)",fontWeight:"600"}}>{health.model.version}</span>
                </div>
              </div>
            </div>

            {/* Recent sessions */}
            <div className="adm-card">
              <h3>Recent Sessions</h3>
              <table className="adm-table">
                <thead><tr><th>Name</th><th>Instances</th><th>Severity</th><th>Date</th></tr></thead>
                <tbody>
                  {stats.recent_sessions.map(s => (
                    <tr key={s.id}>
                      <td>{s.name}</td>
                      <td>{s.total_instances}</td>
                      <td>
                        {s.overall_severity && (
                          <span style={{color:SEV_COLORS[s.overall_severity]||"#64748b",fontWeight:"600",fontSize:"0.75rem"}}>{s.overall_severity}</span>
                        )}
                      </td>
                      <td style={{color:"var(--text-faint)",fontSize:"0.75rem"}}>{new Date(s.created_at).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {/* ── Model tab ── */}
        {!loading && tab==="Model" && model && (
          <>
            {/* Model switcher */}
            <div className="adm-card" style={{marginBottom:"1rem"}}>
              <h3>Model Management</h3>
              <div style={{display:"flex",flexWrap:"wrap",gap:"0.5rem",marginBottom:"0.75rem"}}>
                {model.available_versions.map(v => (
                  <button key={v} onClick={() => handleSwitchModel(v)} disabled={switching || v===model.active_version}
                    style={{
                      padding:"6px 14px", borderRadius:"8px", cursor: v===model.active_version?"default":"pointer",
                      fontSize:"0.82rem", fontWeight:"600", border:"1px solid",
                      background: v===model.active_version ? "var(--accent)22" : "var(--bg-elevated)",
                      color:      v===model.active_version ? "var(--accent)"   : "var(--text-muted)",
                      borderColor:v===model.active_version ? "var(--accent)"   : "var(--border)",
                      opacity: switching ? 0.5 : 1,
                    }}>
                    {v.replace("/best.pt","")} {v===model.active_version ? "✓ Active" : ""}
                  </button>
                ))}
              </div>
              {switching && <p style={{color:"var(--text-muted)",fontSize:"0.82rem"}}>Switching model... this may take a few seconds.</p>}
            </div>

            {/* Metrics per version */}
            {Object.entries(model.metrics).map(([version, m]) => (
              <div key={version} className="adm-card" style={{marginBottom:"1rem"}}>
                <h3>Metrics — {version.replace("/best.pt","")}</h3>
                <p style={{fontSize:"0.75rem",color:"var(--text-muted)",marginBottom:"1rem"}}>
                  {m.epochs} epochs trained · Best at epoch {m.best_epoch}
                </p>

                {/* Key metrics grid */}
                <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(110px,1fr))",gap:"0.6rem",marginBottom:"1.25rem"}}>
                  <MetricBadge label="mAP50 Box"  value={m.best.mAP50_box}  color="#3b82f6" />
                  <MetricBadge label="mAP50 Mask" value={m.best.mAP50_mask} color="#a855f7" />
                  <MetricBadge label="mAP50-95"   value={m.best.mAP50_95_box} color="#06b6d4" />
                  <MetricBadge label="Precision"  value={m.best.precision}  color="#22c55e" />
                  <MetricBadge label="Recall"     value={m.best.recall}     color="#f59e0b" />
                </div>

                {/* Training curves */}
                <p style={{fontSize:"0.72rem",color:"var(--text-faint)",letterSpacing:"0.05em",textTransform:"uppercase",marginBottom:"0.5rem"}}>Training Curve</p>
                <LineChart
                  data={m.curve}
                  lines={[
                    {key:"mAP50_box",  label:"mAP50 Box",  color:"#3b82f6"},
                    {key:"mAP50_mask", label:"mAP50 Mask", color:"#a855f7"},
                    {key:"precision",  label:"Precision",  color:"#22c55e"},
                    {key:"recall",     label:"Recall",     color:"#f59e0b"},
                  ]}
                />
              </div>
            ))}
          </>
        )}

        {/* ── Users tab ── */}
        {!loading && tab==="Users" && users && (
          <div className="adm-card">
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"1rem",flexWrap:"wrap",gap:"0.5rem"}}>
              <h3 style={{margin:0}}>{users.total} Users</h3>
              <input className="adm-input" placeholder="Search users..." value={userSearch}
                onChange={e => setUserSearch(e.target.value)}
                onKeyDown={e => e.key==="Enter" && load("Users")} />
            </div>
            <div style={{overflowX:"auto"}}>
              <table className="adm-table">
                <thead><tr><th>Username</th><th>Email</th><th>Role</th><th>Sessions</th><th>Last Login</th><th>Status</th></tr></thead>
                <tbody>
                  {users.users.map(u => (
                    <tr key={u.id}>
                      <td style={{fontWeight:"500",color:"var(--text-primary)"}}>{u.username}</td>
                      <td>{u.email}</td>
                      <td>
                        {u.role === "admin"
                          ? <span style={{color:"var(--accent)",fontWeight:"700",fontSize:"0.72rem"}}>ADMIN</span>
                          : <span style={{color:"var(--text-faint)",fontSize:"0.75rem"}}>user</span>
                        }
                      </td>
                      <td>{u.sessions}</td>
                      <td style={{color:"var(--text-faint)",fontSize:"0.75rem"}}>{u.last_login ? new Date(u.last_login).toLocaleDateString() : "Never"}</td>
                      <td>
                        {u.role !== "admin" && (
                          <button className="toggle-btn"
                            onClick={() => handleToggleUser(u.id, u.is_active)}
                            style={{
                              background: u.is_active ? "#22c55e18" : "#ef444418",
                              color:      u.is_active ? "#22c55e"   : "#ef4444",
                              borderColor:u.is_active ? "#22c55e44" : "#ef444444",
                            }}>
                            {u.is_active ? "Active" : "Disabled"}
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── Logs tab ── */}
        {!loading && tab==="Logs" && logs && (
          <div className="adm-card">
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"1rem",flexWrap:"wrap",gap:"0.5rem"}}>
              <h3 style={{margin:0}}>{logs.total} Log Entries</h3>
              <div style={{display:"flex",gap:"0.5rem"}}>
                <select className="adm-input" value={logFilter} onChange={e => { setLogFilter(e.target.value); load("Logs"); }}>
                  <option value="">All actions</option>
                  {["login","register","logout","detect","list_sessions","delete_session","update_session","admin_view","admin_action","feedback","token_refresh"].map(a => (
                    <option key={a} value={a}>{a}</option>
                  ))}
                </select>
              </div>
            </div>
            <div style={{overflowX:"auto"}}>
              <table className="adm-table">
                <thead><tr><th>Time</th><th>User</th><th>Action</th><th>Method</th><th>Path</th><th>Status</th><th>IP</th></tr></thead>
                <tbody>
                  {logs.logs.map(l => (
                    <tr key={l.id}>
                      <td style={{color:"var(--text-faint)",fontSize:"0.72rem",whiteSpace:"nowrap"}}>{new Date(l.created_at).toLocaleString()}</td>
                      <td style={{color:"var(--text-primary)",fontWeight:"500"}}>{l.username}</td>
                      <td>
                        <span style={{
                          background:"var(--bg-elevated)",border:"1px solid var(--border)",
                          borderRadius:"5px",padding:"1px 6px",fontSize:"0.72rem",color:"var(--text-secondary)",
                        }}>{l.action}</span>
                      </td>
                      <td style={{fontSize:"0.72rem",color: l.method==="DELETE"?"var(--danger)":l.method==="POST"?"var(--accent)":"var(--text-muted)",fontWeight:"600"}}>{l.method}</td>
                      <td style={{color:"var(--text-faint)",fontSize:"0.72rem",maxWidth:"160px",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{l.path}</td>
                      <td>
                        <span style={{
                          color: l.status < 300 ? "var(--success)" : l.status < 500 ? "var(--warning)" : "var(--danger)",
                          fontWeight:"600",fontSize:"0.75rem",
                        }}>{l.status}</span>
                      </td>
                      <td style={{color:"var(--text-faint)",fontSize:"0.72rem"}}>{l.ip_address}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── Feedback tab ── */}
        {!loading && tab==="Feedback" && feedback && (
          <div className="adm-card">
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"1rem"}}>
              <h3 style={{margin:0}}>{feedback.total} Feedback Entries</h3>
              {feedback.avg_rating && (
                <div style={{display:"flex",alignItems:"center",gap:"0.4rem"}}>
                  <span style={{fontSize:"1.2rem"}}>{"★".repeat(Math.round(feedback.avg_rating))}</span>
                  <span style={{fontSize:"0.82rem",color:"var(--text-muted)"}}>avg {feedback.avg_rating.toFixed(1)}/5</span>
                </div>
              )}
            </div>
            <div style={{display:"flex",flexDirection:"column",gap:"0.6rem"}}>
              {feedback.feedbacks.map(f => (
                <div key={f.id} style={{background:"var(--bg-elevated)",borderRadius:"10px",padding:"0.85rem",border:"1px solid var(--border)"}}>
                  <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"0.4rem"}}>
                    <span style={{fontWeight:"600",color:"var(--text-primary)",fontSize:"0.85rem"}}>{f.username}</span>
                    <div style={{display:"flex",alignItems:"center",gap:"0.5rem"}}>
                      {f.rating && <span style={{color:"#eab308",fontSize:"0.8rem"}}>{"★".repeat(Math.round(f.rating))} {f.rating}/5</span>}
                      <span style={{color:"var(--text-faint)",fontSize:"0.72rem"}}>{new Date(f.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <p style={{color:"var(--text-secondary)",fontSize:"0.83rem",lineHeight:"1.5"}}>{f.message}</p>
                </div>
              ))}
              {feedback.feedbacks.length === 0 && (
                <p style={{color:"var(--text-faint)",fontSize:"0.85rem",textAlign:"center",padding:"1rem"}}>No feedback yet.</p>
              )}
            </div>
          </div>
        )}
      </div>
    </>
  );
}