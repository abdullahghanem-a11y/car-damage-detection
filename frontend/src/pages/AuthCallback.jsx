import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { setAccessToken } from "../services/api";
import { useAuth } from "../context/AuthContext";

export default function AuthCallback() {
  const [searchParams] = useSearchParams();
  const navigate       = useNavigate();
  const { loadUser }   = useAuth();
  const [error, setError] = useState(null);

  useEffect(() => {
    const token = searchParams.get("token");
    const err   = searchParams.get("error");

    if (err) {
      const messages = {
        invalid_state:         "Security check failed. Please try again.",
        token_exchange_failed: "Failed to connect with Google.",
        user_info_failed:      "Could not retrieve your Google profile.",
        missing_user_info:     "Google did not provide required information.",
        account_disabled:      "Your account has been disabled.",
      };
      setError(messages[err] || "Google sign-in failed. Please try again.");
      setTimeout(() => navigate("/login"), 3000);
      return;
    }

    if (!token) {
      setError("No token received.");
      setTimeout(() => navigate("/login"), 3000);
      return;
    }

    // Store token in memory → load user into context → navigate home
    setAccessToken(token);
    loadUser()
      .then(() => navigate("/", { replace: true }))
      .catch(() => {
        setError("Failed to load your profile.");
        setTimeout(() => navigate("/login"), 3000);
      });
  }, []);

  return (
    <div style={{
      minHeight: "100vh", background: "var(--bg-base)",
      display: "flex", alignItems: "center", justifyContent: "center",
      flexDirection: "column", gap: "1rem",
    }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        .cb-spinner { width:36px; height:36px; border:3px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.8s linear infinite; }
      `}</style>

      {error ? (
        <div style={{ textAlign: "center" }}>
          <p style={{ color: "var(--danger)", fontSize: "0.9rem", marginBottom: "0.5rem" }}>{error}</p>
          <p style={{ color: "var(--text-faint)", fontSize: "0.8rem" }}>Redirecting to login...</p>
        </div>
      ) : (
        <>
          <div className="cb-spinner" />
          <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>Signing you in with Google...</p>
        </>
      )}
    </div>
  );
}