import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";

export default function Navbar() {
  const location                  = useLocation();
  const navigate                  = useNavigate();
  const { theme, toggle }         = useTheme();
  const { user, logout, isAdmin } = useAuth();
  const [menuOpen, setMenuOpen]   = useState(false);

  const isActive = (path) => location.pathname === path;

  const handleLogout = async () => {
    setMenuOpen(false);
    await logout();
    navigate("/login");
  };

  const closeMenu = () => setMenuOpen(false);

  const NAV_LINKS = [
    { path: "/",        label: "Detect"  },
    { path: "/history", label: "History" },
    ...(isAdmin ? [{ path: "/admin", label: "Admin Dashboard", admin: true }] : []),
  ];

  return (
    <>
      <style>{`
        /* ── Shared ── */
        .nav-link {
          position: relative; text-decoration: none;
          color: var(--text-muted); font-weight: 500;
          transition: color 0.2s; white-space: nowrap;
        }
        .nav-link:hover { color: var(--text-primary); }
        .nav-link.active { color: var(--text-primary); }

        /* Desktop underline */
        .nav-link-desktop::after {
          content: ''; position: absolute; bottom: -2px; left: 0;
          width: 0; height: 2px;
          background: linear-gradient(90deg, var(--accent), #06b6d4);
          border-radius: 2px; transition: width 0.25s ease;
        }
        .nav-link-desktop.active::after,
        .nav-link-desktop:hover::after { width: 100%; }

        .nav-icon-btn {
          background: var(--bg-elevated); border: 1px solid var(--border);
          border-radius: 8px; width: 32px; height: 32px;
          display: flex; align-items: center; justify-content: center;
          cursor: pointer; transition: all 0.2s; flex-shrink: 0;
        }
        .nav-icon-btn:hover { background: var(--bg-hover); border-color: var(--text-faint); }

        /* ── Desktop nav (>600px) ── */
        .nav-desktop {
          display: flex; align-items: center; gap: 1.25rem;
        }
        .nav-link-desktop { font-size: 0.88rem; padding: 4px 0; }

        .username-chip {
          font-size: 0.72rem; color: var(--text-faint);
          max-width: 90px; overflow: hidden;
          text-overflow: ellipsis; white-space: nowrap;
        }
        .logout-btn-desk {
          background: var(--bg-elevated); border: 1px solid var(--border);
          border-radius: 8px; height: 30px; padding: 0 10px;
          cursor: pointer; font-size: 0.75rem; color: var(--text-muted);
          transition: all 0.15s; white-space: nowrap;
          display: flex; align-items: center; gap: 4px;
        }
        .logout-btn-desk:hover { border-color: var(--danger); color: var(--danger); }

        .admin-badge {
          background: var(--accent)22; color: var(--accent);
          border: 1px solid var(--accent)44; border-radius: 6px;
          padding: 2px 8px; font-size: 0.68rem; font-weight: 700;
          letter-spacing: 0.05em; text-decoration: none; flex-shrink: 0;
          transition: background 0.15s;
        }
        .admin-badge:hover { background: var(--accent)33; }

        /* ── Hamburger (≤600px) ── */
        .hamburger {
          display: none; flex-direction: column; justify-content: space-between;
          width: 22px; height: 16px; cursor: pointer; padding: 0;
          background: none; border: none; flex-shrink: 0;
        }
        .hamburger span {
          display: block; height: 2px; width: 100%;
          background: var(--text-primary); border-radius: 2px;
          transition: all 0.25s ease; transform-origin: center;
        }
        .hamburger.open span:nth-child(1) { transform: translateY(7px) rotate(45deg); }
        .hamburger.open span:nth-child(2) { opacity: 0; transform: scaleX(0); }
        .hamburger.open span:nth-child(3) { transform: translateY(-7px) rotate(-45deg); }

        /* ── Mobile menu drawer ── */
        .mobile-menu {
          display: none;
          position: absolute; top: 52px; left: 0; right: 0;
          background: var(--bg-surface); border-bottom: 1px solid var(--border);
          box-shadow: var(--shadow); z-index: 99;
          flex-direction: column;
          overflow: hidden;
          max-height: 0;
          transition: max-height 0.3s ease;
        }
        .mobile-menu.open { max-height: 500px; }

        .mobile-link {
          display: flex; align-items: center; gap: 0.75rem;
          padding: 0.85rem 1rem; text-decoration: none;
          color: var(--text-secondary); font-size: 0.92rem; font-weight: 500;
          border-bottom: 1px solid var(--border-subtle);
          transition: background 0.15s;
        }
        .mobile-link:hover { background: var(--bg-elevated); }
        .mobile-link.active { color: var(--accent); }
        .mobile-link.admin { color: var(--accent); }

        .mobile-footer {
          padding: 0.85rem 1rem;
          display: flex; justify-content: space-between; align-items: center;
        }
        .mobile-logout {
          background: var(--bg-elevated); border: 1px solid var(--border);
          border-radius: 8px; padding: 6px 14px; cursor: pointer;
          font-size: 0.82rem; color: var(--text-muted); transition: all 0.15s;
        }
        .mobile-logout:hover { border-color: var(--danger); color: var(--danger); }

        /* ── Overlay ── */
        .menu-overlay {
          display: none; position: fixed; inset: 0; z-index: 98;
          background: transparent;
        }

        @media (max-width: 600px) {
          .nav-desktop  { display: none; }
          .hamburger    { display: flex; }
          .mobile-menu  { display: flex; }
          .menu-overlay.open { display: block; }
        }
      `}</style>

      {/* Overlay to close menu on outside tap */}
      {menuOpen && (
        <div className="menu-overlay open" onClick={closeMenu} />
      )}

      <nav style={{
        background:     "var(--bg-surface)",
        padding:        "0 1rem",
        height:         "52px",
        display:        "flex",
        alignItems:     "center",
        justifyContent: "space-between",
        borderBottom:   "1px solid var(--border)",
        position:       "sticky",
        top:            0,
        zIndex:         100,
        boxShadow:      "var(--shadow)",
        transition:     "background 0.25s ease",
        gap:            "0.5rem",
      }}>

        {/* Logo */}
        <Link to="/" onClick={closeMenu} style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: "0.4rem", flexShrink: 0 }}>
          <img src="/favicon.png" alt="AUTOPS" style={{
            width: "28px", height: "28px", objectFit: "contain",
            filter: theme === "dark" ? "invert(1) brightness(2)" : "none",
            transition: "filter 0.25s ease",
          }} />
          <span style={{ fontSize: "0.92rem", fontWeight: "700", color: "var(--text-primary)", letterSpacing: "0.02em" }}>
            AUT<span style={{ color: "var(--accent)" }}>O_O</span>PS
          </span>
        </Link>

        {/* ── Desktop nav ── */}
        <div className="nav-desktop">
          {NAV_LINKS.map(({ path, label, admin }) => (
            <Link key={path} to={path}
              className={`nav-link nav-link-desktop ${isActive(path) ? "active" : ""} ${admin ? "admin-badge" : ""}`}>
              {label}
            </Link>
          ))}

          {user && <span className="username-chip">{user.username}</span>}

          {user && (
            <button className="logout-btn-desk" onClick={handleLogout}>
              ⏻ Logout
            </button>
          )}

          <button className="nav-icon-btn" onClick={toggle} title="Toggle theme">
            <img
              src={theme === "dark" ? "/lightIcone.png" : "/darkIcone.png"}
              alt="toggle theme"
              style={{ width: "16px", height: "16px", objectFit: "contain", filter: theme === "dark" ? "invert(1) brightness(2)" : "none" }}
            />
          </button>
        </div>

        {/* ── Mobile: hamburger only ── */}
        <button
          className={`hamburger ${menuOpen ? "open" : ""}`}
          onClick={() => setMenuOpen(o => !o)}
          aria-label="Toggle menu"
        >
          <span /><span /><span />
        </button>
      </nav>

      {/* ── Mobile dropdown menu ── */}
      <div className={`mobile-menu ${menuOpen ? "open" : ""}`}>

        {NAV_LINKS.map(({ path, label, admin }) => (
          <Link key={path} to={path} onClick={closeMenu}
            className={`mobile-link ${isActive(path) ? "active" : ""} ${admin ? "admin" : ""}`}>
            {admin ? "⚙️" : isActive(path) ? "→" : "·"} {label}
          </Link>
        ))}

        <div className="mobile-footer">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            {user && (
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                👤 {user.username}
              </span>
            )}
            {isAdmin && (
              <span style={{ background: "var(--accent)22", color: "var(--accent)", border: "1px solid var(--accent)44", borderRadius: "6px", padding: "1px 7px", fontSize: "0.68rem", fontWeight: "700" }}>
                ADMIN
              </span>
            )}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            {/* Theme toggle in mobile footer */}
            <button className="nav-icon-btn" onClick={toggle} title="Toggle theme">
              <img
                src={theme === "dark" ? "/lightIcone.png" : "/darkIcone.png"}
                alt="toggle theme"
                style={{ width: "16px", height: "16px", objectFit: "contain", filter: theme === "dark" ? "invert(1) brightness(2)" : "none" }}
              />
            </button>

            {user && (
              <button className="mobile-logout" onClick={handleLogout}>⏻ Logout</button>
            )}
          </div>
        </div>
      </div>
    </>
  );
}