import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar  from "./components/Navbar";
import Home    from "./pages/Home";
import History from "./pages/History";

export default function App() {
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Navbar />
      <Routes>
        <Route path="/"        element={<Home />}    />
        <Route path="/history" element={<History />} />
      </Routes>
    </BrowserRouter>
  );
}