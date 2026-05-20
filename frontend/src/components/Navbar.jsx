import { Link, useLocation } from "react-router-dom";
import { useTheme } from "../context/ThemeContext";

export default function Navbar() {
  const location           = useLocation();
  const { theme, toggle }  = useTheme();
  const isActive           = (path) => location.pathname === path;

  return (
    <>
      <style>{`
        .nav-link {
          position: relative;
          text-decoration: none;
          color: var(--text-muted);
          font-size: 0.9rem;
          font-weight: 500;
          padding: 4px 0;
          letter-spacing: 0.04em;
          transition: color 0.2s;
        }
        .nav-link:hover { color: var(--text-primary); }
        .nav-link.active { color: var(--text-primary); }
        .nav-link::after {
          content: '';
          position: absolute;
          bottom: -2px; left: 0;
          width: 0; height: 2px;
          background: linear-gradient(90deg, var(--accent), #06b6d4);
          border-radius: 2px;
          transition: width 0.25s ease;
        }
        .nav-link.active::after,
        .nav-link:hover::after { width: 100%; }

        .theme-btn {
          background: var(--bg-elevated);
          border: 1px solid var(--border);
          border-radius: 8px;
          width: 32px; height: 32px;
          display: flex; align-items: center; justify-content: center;
          cursor: pointer;
          transition: background 0.2s, border-color 0.2s;
          flex-shrink: 0;
        }
        .theme-btn:hover { background: var(--bg-hover); border-color: var(--text-faint); }

        .nav-right { display: flex; align-items: center; gap: 1.5rem; }

        @media (max-width: 400px) {
          .nav-right { gap: 1rem; }
          .nav-link { font-size: 0.82rem; }
        }
      `}</style>

      <nav style={{
        background:     "var(--bg-surface)",
        padding:        "0 1rem",
        height:         "56px",
        display:        "flex",
        alignItems:     "center",
        justifyContent: "space-between",
        borderBottom:   "1px solid var(--border)",
        position:       "sticky",
        top:            0,
        zIndex:         100,
        boxShadow:      "var(--shadow)",
        transition:     "background 0.25s ease",
      }}>

        {/* Logo */}
        <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: "0.5rem", flexShrink: 0 }}>
          <img
            src="/favicon.png"
            alt="AUTOPS"
            style={{
              width: "30px", height: "30px", objectFit: "contain",
              filter:    theme === "dark" ? "invert(1) brightness(2)" : "none",
              transition: "filter 0.25s ease",
            }}
          />
          <span style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--text-primary)", letterSpacing: "0.02em" }}>
            AUT<span style={{ color: "var(--accent)" }}>O_O</span>PS
          </span>
        </Link>

        {/* Right: links + toggle */}
        <div className="nav-right">
          {[{ path: "/", label: "Detect" }, { path: "/history", label: "History" }].map(({ path, label }) => (
            <Link key={path} to={path} className={`nav-link ${isActive(path) ? "active" : ""}`}>
              {label}
            </Link>
          ))}
          <button className="theme-btn" onClick={toggle} title="Toggle theme">
            <img
              src={theme === "dark" ? "/lightIcone.png" : "/darkIcone.png"}
              alt="toggle theme"
              style={{
                width: "18px", height: "18px", objectFit: "contain",
                filter: theme === "dark" ? "invert(1) brightness(2)" : "none",
                transition: "filter 0.25s ease",
              }}
            />
          </button>
        </div>
      </nav>
    </>
  );
}