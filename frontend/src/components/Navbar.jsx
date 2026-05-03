import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const location = useLocation();

  const isActive = (path) =>
    location.pathname === path
      ? { borderBottom: "2px solid #3b82f6", color: "#3b82f6" }
      : {};

  return (
    <nav style={{
      background:     "#1e293b",
      padding:        "0 2rem",
      height:         "64px",
      display:        "flex",
      alignItems:     "center",
      justifyContent: "space-between",
      borderBottom:   "1px solid #334155",
      position:       "sticky",
      top:            0,
      zIndex:         100,
    }}>
      {/* Logo */}
      <Link to="/" style={{ textDecoration: "none" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span style={{ fontSize: "1.5rem" }}>🚗</span>
          <span style={{
            fontSize:   "1.1rem",
            fontWeight: "700",
            color:      "#f1f5f9",
          }}>
            CarDD
          </span>
        </div>
      </Link>

      {/* Links */}
      <div style={{ display: "flex", gap: "2rem" }}>
        {[
          { path: "/",        label: "Detect"  },
          { path: "/history", label: "History" },
        ].map(({ path, label }) => (
          <Link
            key={path}
            to={path}
            style={{
              textDecoration: "none",
              color:          "#94a3b8",
              fontSize:       "0.95rem",
              fontWeight:     "500",
              paddingBottom:  "4px",
              ...isActive(path),
            }}
          >
            {label}
          </Link>
        ))}
      </div>
    </nav>
  );
}