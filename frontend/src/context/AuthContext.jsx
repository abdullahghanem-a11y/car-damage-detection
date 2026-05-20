import { createContext, useContext, useState, useEffect } from "react";
import {
  login    as apiLogin,
  logout   as apiLogout,
  register as apiRegister,
  refreshAccessToken,
  getMe,
} from "../services/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user,    setUser]    = useState(null);
  const [loading, setLoading] = useState(true);

  // On mount — try to restore session via httpOnly refresh cookie
  useEffect(() => {
    refreshAccessToken()
      .then(() => getMe())
      .then((userData) => setUser(userData))
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

  const login = async (usernameOrEmail, password) => {
    const data = await apiLogin(usernameOrEmail, password);
    setUser(data.user);
    return data;
  };

  const register = async (email, username, password) => {
    const data = await apiRegister(email, username, password);
    setUser(data.user);
    return data;
  };

  const logout = async () => {
    await apiLogout();
    setUser(null);
  };

  // Called after Google OAuth callback sets token in memory
  const loadUser = async () => {
    const userData = await getMe();
    setUser(userData);
    return userData;
  };

  const isAdmin = user?.role === "admin";

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, loadUser, isAdmin }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
};