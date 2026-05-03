import axios from "axios";

const API = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
});

// ── Detection ──────────────────────────────────
export const detectDamage = async (files) => {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await API.post("/api/detect/", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

// ── History ────────────────────────────────────
export const getHistory = async (skip = 0, limit = 20) => {
  const response = await API.get(`/api/detect/history?skip=${skip}&limit=${limit}`);
  return response.data;
};

export const getDetectionById = async (id) => {
  const response = await API.get(`/api/detect/history/${id}`);
  return response.data;
};