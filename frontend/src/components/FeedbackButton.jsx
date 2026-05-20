/**
 * FeedbackButton.jsx
 *
 * Floating feedback button visible on all pages.
 * Opens a modal with a message field + star rating.
 * Submits to POST /api/auth/feedback.
 */

import { useState } from "react";
import { submitFeedback } from "../services/api";
import { useAuth } from "../context/AuthContext";

// ── Star rating ────────────────────────────────────────────────────────────
function StarRating({ value, onChange }) {
  const [hovered, setHovered] = useState(0);
  return (
    <div style={{ display: "flex", gap: "0.25rem" }}>
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          type="button"
          onClick={() => onChange(star === value ? 0 : star)}
          onMouseEnter={() => setHovered(star)}
          onMouseLeave={() => setHovered(0)}
          style={{
            background: "none", border: "none", cursor: "pointer",
            fontSize: "1.6rem", padding: "0 2px",
            color: star <= (hovered || value) ? "#eab308" : "var(--border)",
            transition: "color 0.15s, transform 0.1s",
            transform: star <= (hovered || value) ? "scale(1.15)" : "scale(1)",
          }}
        >
          ★
        </button>
      ))}
      {value > 0 && (
        <span style={{ fontSize: "0.78rem", color: "var(--text-faint)", alignSelf: "center", marginLeft: "4px" }}>
          {["", "Poor", "Fair", "Good", "Great", "Excellent"][value]}
        </span>
      )}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────
export default function FeedbackButton() {
  const { user } = useAuth();

  const [open,       setOpen]       = useState(false);
  const [message,    setMessage]    = useState("");
  const [rating,     setRating]     = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [submitted,  setSubmitted]  = useState(false);
  const [error,      setError]      = useState(null);

  // Don't show for admins or unauthenticated users
  if (!user || user.role === "admin") return null;

  const handleSubmit = async () => {
    if (!message.trim()) { setError("Please write a message."); return; }
    setError(null);
    setSubmitting(true);
    try {
      await submitFeedback(message.trim(), rating || null);
      setSubmitted(true);
      setTimeout(() => {
        setOpen(false);
        setSubmitted(false);
        setMessage("");
        setRating(0);
      }, 2000);
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to submit. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleClose = () => {
    if (submitting) return;
    setOpen(false);
    setError(null);
  };

  return (
    <>
      <style>{`
        .fb-btn {
          position: fixed;
          bottom: 1.25rem;
          right: 1.25rem;
          z-index: 500;
          background: linear-gradient(135deg, var(--accent), #0ea5e9);
          color: white;
          border: none;
          border-radius: 50px;
          padding: 0.55rem 1rem;
          font-size: 0.82rem;
          font-weight: 600;
          cursor: pointer;
          box-shadow: 0 4px 16px rgba(59,130,246,0.4);
          display: flex;
          align-items: center;
          gap: 0.4rem;
          transition: transform 0.2s, box-shadow 0.2s;
          letter-spacing: 0.02em;
        }
        .fb-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(59,130,246,0.5);
        }
        .fb-overlay {
          position: fixed; inset: 0; z-index: 600;
          background: #00000077; backdrop-filter: blur(4px);
          display: flex; align-items: flex-end; justify-content: center;
          padding: 1rem;
          animation: fbFadeIn 0.15s ease;
        }
        @keyframes fbFadeIn { from { opacity: 0; } to { opacity: 1; } }
        .fb-modal {
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: 16px;
          width: 100%; max-width: 420px;
          padding: 1.5rem;
          box-shadow: var(--shadow);
          animation: fbSlideUp 0.2s ease;
        }
        @keyframes fbSlideUp {
          from { transform: translateY(20px); opacity: 0; }
          to   { transform: translateY(0);    opacity: 1; }
        }
        .fb-textarea {
          width: 100%;
          background: var(--bg-elevated);
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 0.75rem;
          color: var(--text-primary);
          font-size: 0.88rem;
          font-family: inherit;
          resize: vertical;
          min-height: 100px;
          outline: none;
          transition: border-color 0.2s;
          box-sizing: border-box;
          line-height: 1.5;
        }
        .fb-textarea:focus { border-color: var(--accent); }
        .fb-textarea::placeholder { color: var(--text-faint); }
        .fb-submit {
          width: 100%;
          background: linear-gradient(135deg, var(--accent), #0ea5e9);
          color: white; border: none; border-radius: 10px;
          padding: 0.7rem; font-size: 0.9rem; font-weight: 600;
          cursor: pointer; transition: opacity 0.2s;
        }
        .fb-submit:hover:not(:disabled) { opacity: 0.9; }
        .fb-submit:disabled { opacity: 0.5; cursor: not-allowed; }
        .fb-cancel {
          width: 100%;
          background: transparent; color: var(--text-muted);
          border: 1px solid var(--border); border-radius: 10px;
          padding: 0.6rem; font-size: 0.85rem; cursor: pointer;
          transition: all 0.15s;
        }
        .fb-cancel:hover { border-color: var(--text-faint); color: var(--text-primary); }
        .char-count {
          font-size: 0.7rem;
          color: ${message.length > 900 ? "var(--danger)" : "var(--text-faint)"};
          text-align: right;
          margin-top: 4px;
        }
      `}</style>

      {/* Floating button */}
      <button className="fb-btn" onClick={() => setOpen(true)}>
        💬 Feedback
      </button>

      {/* Modal */}
      {open && (
        <div className="fb-overlay" onClick={handleClose}>
          <div className="fb-modal" onClick={(e) => e.stopPropagation()}>

            {submitted ? (
              /* Success state */
              <div style={{ textAlign: "center", padding: "1rem 0" }}>
                <div style={{ fontSize: "2.5rem", marginBottom: "0.75rem" }}>🎉</div>
                <h3 style={{ color: "var(--text-primary)", fontSize: "1rem", fontWeight: "700", marginBottom: "0.4rem" }}>
                  Thank you!
                </h3>
                <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                  Your feedback has been submitted.
                </p>
              </div>
            ) : (
              <>
                {/* Header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1.25rem" }}>
                  <div>
                    <h3 style={{ color: "var(--text-primary)", fontSize: "1rem", fontWeight: "700", marginBottom: "2px" }}>
                      Share your feedback
                    </h3>
                    <p style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>
                      Help us improve AUTOPS
                    </p>
                  </div>
                  <button onClick={handleClose} style={{
                    background: "var(--bg-elevated)", border: "1px solid var(--border)",
                    borderRadius: "8px", padding: "4px 9px", cursor: "pointer",
                    color: "var(--text-muted)", fontSize: "0.85rem",
                  }}>✕</button>
                </div>

                {/* Star rating */}
                <div style={{ marginBottom: "1rem" }}>
                  <p style={{ fontSize: "0.75rem", color: "var(--text-faint)", letterSpacing: "0.04em", textTransform: "uppercase", marginBottom: "0.5rem" }}>
                    Rating (optional)
                  </p>
                  <StarRating value={rating} onChange={setRating} />
                </div>

                {/* Message */}
                <div style={{ marginBottom: "1rem" }}>
                  <p style={{ fontSize: "0.75rem", color: "var(--text-faint)", letterSpacing: "0.04em", textTransform: "uppercase", marginBottom: "0.5rem" }}>
                    Message
                  </p>
                  <textarea
                    className="fb-textarea"
                    placeholder="Tell us what you think — what works well, what could be better..."
                    value={message}
                    maxLength={1000}
                    onChange={(e) => setMessage(e.target.value)}
                  />
                  <p className="char-count">{message.length}/1000</p>
                </div>

                {/* Error */}
                {error && (
                  <div style={{
                    background: "var(--bg-elevated)", border: "1px solid var(--danger)",
                    borderRadius: "8px", padding: "0.6rem 0.85rem",
                    color: "var(--danger)", fontSize: "0.82rem", marginBottom: "1rem",
                  }}>
                    {error}
                  </div>
                )}

                {/* Actions */}
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  <button className="fb-submit" onClick={handleSubmit} disabled={submitting || !message.trim()}>
                    {submitting ? "Submitting..." : "Submit Feedback"}
                  </button>
                  <button className="fb-cancel" onClick={handleClose}>Cancel</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}