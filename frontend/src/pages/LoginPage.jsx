import { useState } from "react";
import { useNavigate, Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function LoginPage() {
  const { user, login, register, loading } = useAuth();
  const navigate = useNavigate();

  const [mode,     setMode]     = useState("login");   // "login" | "register"
  const [form,     setForm]     = useState({ email: "", username: "", password: "", usernameOrEmail: "" });
  const [error,    setError]    = useState(null);
  const [submitting, setSubmitting] = useState(false);

  // Already logged in — redirect
  if (!loading && user) return <Navigate to="/" replace />;

  const set = (field) => (e) => setForm(f => ({ ...f, [field]: e.target.value }));

  const handleSubmit = async () => {
    setError(null);
    setSubmitting(true);
    try {
      if (mode === "login") {
        await login(form.usernameOrEmail, form.password);
      } else {
        if (!form.email || !form.username || !form.password) {
          setError("All fields are required"); return;
        }
        if (form.password.length < 8) {
          setError("Password must be at least 8 characters"); return;
        }
        await register(form.email, form.username, form.password);
      }
      navigate("/");
    } catch (err) {
      setError(err.response?.data?.detail || "Something went wrong. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <style>{`
        .auth-wrap {
          min-height: 100vh; background: var(--bg-base);
          display: flex; align-items: center; justify-content: center;
          padding: 1rem;
        }
        .auth-card {
          background: var(--bg-surface); border: 1px solid var(--border);
          border-radius: 16px; padding: 2rem; width: 100%; max-width: 400px;
          box-shadow: var(--shadow);
        }
        .auth-input {
          width: 100%; background: var(--bg-elevated);
          border: 1px solid var(--border); border-radius: 8px;
          padding: 0.65rem 0.85rem; color: var(--text-primary);
          font-size: 0.9rem; outline: none; transition: border-color 0.2s;
          box-sizing: border-box;
        }
        .auth-input:focus { border-color: var(--accent); }
        .auth-input::placeholder { color: var(--text-faint); }
        .auth-btn {
          width: 100%; background: linear-gradient(135deg, var(--accent), #0ea5e9);
          color: white; border: none; border-radius: 8px;
          padding: 0.75rem; font-size: 0.92rem; font-weight: 600;
          cursor: pointer; transition: opacity 0.2s;
        }
        .auth-btn:hover:not(:disabled) { opacity: 0.9; }
        .auth-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .auth-tab {
          flex: 1; padding: 0.6rem; background: none; border: none;
          cursor: pointer; font-size: 0.88rem; color: var(--text-muted);
          border-bottom: 2px solid transparent; transition: all 0.15s;
        }
        .auth-tab.on { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
        .field-label {
          font-size: 0.75rem; color: var(--text-muted);
          margin-bottom: 5px; display: block;
        }
      `}</style>

      <div className="auth-wrap">
        <div className="auth-card">

          {/* Logo */}
          <div style={{ textAlign: "center", marginBottom: "1.75rem" }}>
            <img src="/favicon.png" alt="AUTOPS" style={{ width: "48px", height: "48px", objectFit: "contain", filter: "invert(1) brightness(2)", marginBottom: "0.75rem" }} />
            <h1 style={{ fontSize: "1.4rem", fontWeight: "700", color: "var(--text-primary)" }}>
              AUT<span style={{ color: "var(--accent)" }}>O_O</span>PS
            </h1>
            <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", marginTop: "0.25rem" }}>
              Car Damage Detection
            </p>
          </div>

          {/* Tabs */}
          <div style={{ display: "flex", borderBottom: "1px solid var(--border)", marginBottom: "1.5rem" }}>
            <button className={`auth-tab ${mode==="login"?"on":""}`}   onClick={() => { setMode("login");    setError(null); }}>Sign In</button>
            <button className={`auth-tab ${mode==="register"?"on":""}`} onClick={() => { setMode("register"); setError(null); }}>Register</button>
          </div>

          {/* Fields */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>

            {mode === "register" && (
              <div>
                <label className="field-label">Email</label>
                <input className="auth-input" type="email" placeholder="you@example.com"
                  value={form.email} onChange={set("email")}
                  onKeyDown={(e) => e.key === "Enter" && handleSubmit()} />
              </div>
            )}

            {mode === "register" && (
              <div>
                <label className="field-label">Username</label>
                <input className="auth-input" type="text" placeholder="your_username"
                  value={form.username} onChange={set("username")}
                  onKeyDown={(e) => e.key === "Enter" && handleSubmit()} />
              </div>
            )}

            {mode === "login" && (
              <div>
                <label className="field-label">Email or Username</label>
                <input className="auth-input" type="text" placeholder="Email or username"
                  value={form.usernameOrEmail} onChange={set("usernameOrEmail")}
                  onKeyDown={(e) => e.key === "Enter" && handleSubmit()} />
              </div>
            )}

            <div>
              <label className="field-label">Password</label>
              <input className="auth-input" type="password"
                placeholder={mode === "register" ? "Min. 8 characters" : "Password"}
                value={form.password} onChange={set("password")}
                onKeyDown={(e) => e.key === "Enter" && handleSubmit()} />
            </div>

            {/* Error */}
            {error && (
              <div style={{ background: "var(--bg-elevated)", border: "1px solid var(--danger)", borderRadius: "8px", padding: "0.65rem 0.85rem", color: "var(--danger)", fontSize: "0.82rem" }}>
                {error}
              </div>
            )}

            <button className="auth-btn" onClick={handleSubmit} disabled={submitting}>
              {submitting ? "..." : mode === "login" ? "Sign In" : "Create Account"}
            </button>

            {/* Google OAuth placeholder */}
            <div style={{ position: "relative", textAlign: "center" }}>
              <div style={{ position: "absolute", top: "50%", left: 0, right: 0, height: "1px", background: "var(--border)" }} />
              <span style={{ position: "relative", background: "var(--bg-surface)", padding: "0 0.75rem", fontSize: "0.75rem", color: "var(--text-faint)" }}>or</span>
            </div>

            <button
              onClick={() => window.location.href = `${import.meta.env.VITE_API_URL || "http://localhost:8000"}/api/auth/google`}
              style={{ width: "100%", background: "var(--bg-elevated)", color: "var(--text-secondary)", border: "1px solid var(--border)", borderRadius: "8px", padding: "0.65rem", fontSize: "0.88rem", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem", transition: "border-color 0.2s" }}>
              <svg width="18" height="18" viewBox="0 0 48 48">
                <path fill="#FFC107" d="M43.6 20H24v8h11.3C33.7 33.1 29.3 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.1 7.9 3l5.7-5.7C34.1 6.5 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20c11 0 19.6-8 19.6-20 0-1.3-.1-2.7-.4-4z"/>
                <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.5 16 19 13 24 13c3.1 0 5.8 1.1 7.9 3l5.7-5.7C34.1 6.5 29.3 4 24 4c-7.7 0-14.4 4.4-17.7 10.7z"/>
                <path fill="#4CAF50" d="M24 44c5.2 0 9.9-1.9 13.4-5l-6.2-5.2C29.3 35.3 26.8 36 24 36c-5.2 0-9.7-3-11.3-7.3l-6.5 5C9.5 39.6 16.2 44 24 44z"/>
                <path fill="#1976D2" d="M43.6 20H24v8h11.3c-.9 2.4-2.5 4.4-4.7 5.8l6.2 5.2C41 35.6 44 30.2 44 24c0-1.3-.1-2.7-.4-4z"/>
              </svg>
              Continue with Google
            </button>
          </div>
        </div>
      </div>
    </>
  );
}