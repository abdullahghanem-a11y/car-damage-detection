import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider }   from "./context/AuthContext";
import ProtectedRoute     from "./components/ProtectedRoute";
import Navbar             from "./components/Navbar";
import FeedbackButton     from "./components/FeedbackButton";
import Home               from "./pages/Home";
import History            from "./pages/History";
import LoginPage          from "./pages/LoginPage";
import AdminDashboard     from "./pages/AdminDashboard";
import AuthCallback       from "./pages/AuthCallback";

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Routes>
          {/* Public */}
          <Route path="/login"         element={<LoginPage />}    />
          <Route path="/auth/callback" element={<AuthCallback />} />

          {/* Protected — all authenticated users */}
          <Route path="/" element={
            <ProtectedRoute>
              <Navbar />
              <Home />
              <FeedbackButton />
            </ProtectedRoute>
          } />
          <Route path="/history" element={
            <ProtectedRoute>
              <Navbar />
              <History />
              <FeedbackButton />
            </ProtectedRoute>
          } />

          {/* Admin only — no feedback button for admin */}
          <Route path="/admin" element={
            <ProtectedRoute adminOnly>
              <Navbar />
              <AdminDashboard />
            </ProtectedRoute>
          } />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}