/**
 * ProtectedRoute.jsx
 * 
 * Wraps routes that require authentication.
 * - Shows loading spinner while checking session
 * - Redirects to /login if not authenticated
 * - Redirects to / if authenticated but wrong role (adminOnly)
 */

import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function ProtectedRoute({ children, adminOnly = false }) {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div style={{
        height:          "100vh",
        display:         "flex",
        alignItems:      "center",
        justifyContent:  "center",
        background:      "var(--bg-base)",
        flexDirection:   "column",
        gap:             "1rem",
      }}>
        <style>{`
          @keyframes spin { to { transform: rotate(360deg); } }
          .spinner {
            width: 36px; height: 36px;
            border: 3px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
          }
        `}</style>
        <div className="spinner" />
        <p style={{ color: "var(--text-faint)", fontSize: "0.85rem" }}>Loading...</p>
      </div>
    );
  }

  if (!user) return <Navigate to="/login" replace />;

  if (adminOnly && user.role !== "admin") return <Navigate to="/" replace />;

  return children;
}