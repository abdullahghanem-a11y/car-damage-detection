import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL:         BASE_URL,
  withCredentials: true,   // send httpOnly cookies automatically
});

// ── Token stored in memory only (not localStorage) ─────────────────────────
let _accessToken = null;

export function setAccessToken(token) { _accessToken = token; }
export function getAccessToken()      { return _accessToken;  }
export function clearAccessToken()    { _accessToken = null;  }

// ── Axios interceptor — attach access token to every request ──────────────
api.interceptors.request.use((config) => {
  if (_accessToken) {
    config.headers.Authorization = `Bearer ${_accessToken}`;
  }
  return config;
});

// ── Axios interceptor — auto-refresh on 401 ───────────────────────────────
let _refreshing = false;
let _queue      = [];

api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const original = error.config;

    // If 401 and not already retrying and not a refresh/login request
    if (
      error.response?.status === 401 &&
      !original._retry &&
      !original.url.includes("/auth/refresh") &&
      !original.url.includes("/auth/login")
    ) {
      if (_refreshing) {
        // Queue requests while refreshing
        return new Promise((resolve, reject) => {
          _queue.push({ resolve, reject });
        }).then((token) => {
          original.headers.Authorization = `Bearer ${token}`;
          return api(original);
        });
      }

      original._retry = true;
      _refreshing     = true;

      try {
        const { data } = await api.post("/api/auth/refresh");
        setAccessToken(data.access_token);
        _queue.forEach(({ resolve }) => resolve(data.access_token));
        _queue      = [];
        _refreshing = false;
        original.headers.Authorization = `Bearer ${data.access_token}`;
        return api(original);
      } catch (refreshErr) {
        _queue.forEach(({ reject }) => reject(refreshErr));
        _queue      = [];
        _refreshing = false;
        clearAccessToken();
        // Redirect to login
        window.location.href = "/login";
        return Promise.reject(refreshErr);
      }
    }

    return Promise.reject(error);
  }
);


// ── Auth ───────────────────────────────────────────────────────────────────

export async function register(email, username, password) {
  const { data } = await api.post("/api/auth/register", { email, username, password });
  setAccessToken(data.access_token);
  return data;
}

export async function login(username_or_email, password) {
  const { data } = await api.post("/api/auth/login", { username_or_email, password });
  setAccessToken(data.access_token);
  return data;
}

export async function logout() {
  await api.post("/api/auth/logout");
  clearAccessToken();
}

export async function refreshAccessToken() {
  const { data } = await api.post("/api/auth/refresh");
  setAccessToken(data.access_token);
  return data;
}

export async function getMe() {
  const { data } = await api.get("/api/auth/me");
  return data;
}

export async function submitFeedback(message, rating) {
  const { data } = await api.post("/api/auth/feedback", { message, rating });
  return data;
}


// ── Detection ──────────────────────────────────────────────────────────────

export async function detectDamage(files, sessionName) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  const { data } = await api.post(
    `/api/detect/?session_name=${encodeURIComponent(sessionName)}`,
    formData,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return data;
}


// ── Sessions ───────────────────────────────────────────────────────────────

export async function getSessions({ skip=0, limit=20, search="", severity="", sort="date_desc" } = {}) {
  const params = new URLSearchParams({ skip, limit, sort });
  if (search)   params.append("search",   search);
  if (severity) params.append("severity", severity);
  const { data } = await api.get(`/api/sessions/?${params}`);
  return data;
}

export async function getSession(sessionId) {
  const { data } = await api.get(`/api/sessions/${sessionId}`);
  return data;
}

export async function renameSession(sessionId, name) {
  const { data } = await api.patch(`/api/sessions/${sessionId}/rename`, { name });
  return data;
}

export async function deleteSession(sessionId) {
  const { data } = await api.delete(`/api/sessions/${sessionId}`);
  return data;
}

export async function deleteSessionImages(sessionId, detectionIds) {
  const { data } = await api.delete(`/api/sessions/${sessionId}/images`, {
    data: { detection_ids: detectionIds },
  });
  return data;
}

export async function updateDetectionImage(sessionId, detectionId, imageData) {
  const { data } = await api.patch(
    `/api/sessions/${sessionId}/images/${detectionId}`,
    { image_data: imageData }
  );
  return data;
}

export async function reDetectImage(sessionId, detectionId) {
  const { data } = await api.post(`/api/sessions/${sessionId}/images/${detectionId}/redetect`);
  return data;
}

export async function copyDetectionImage(sessionId, detectionId, imageData, copyName) {
  const { data } = await api.post(
    `/api/sessions/${sessionId}/images/${detectionId}/copy`,
    { image_data: imageData, copy_name: copyName }
  );
  return data;
}

export function getRawImageUrl(sessionId, detectionId) {
  return `${BASE_URL}/api/sessions/${sessionId}/images/${detectionId}/raw`;
}

export async function getHistory(skip = 0, limit = 20) {
  return getSessions({ skip, limit });
}


// ── Admin ──────────────────────────────────────────────────────────────────

export async function getAdminStats() {
  const { data } = await api.get("/api/admin/stats");
  return data;
}

export async function getAdminHealth() {
  const { data } = await api.get("/api/admin/health");
  return data;
}

export async function getAdminUsers({ skip=0, limit=20, search="" } = {}) {
  const params = new URLSearchParams({ skip, limit });
  if (search) params.append("search", search);
  const { data } = await api.get(`/api/admin/users?${params}`);
  return data;
}

export async function toggleUser(userId, isActive) {
  const { data } = await api.patch(`/api/admin/users/${userId}`, { is_active: isActive });
  return data;
}

export async function getAdminLogs({ skip=0, limit=50, action="", userId=null, status=null } = {}) {
  const params = new URLSearchParams({ skip, limit });
  if (action) params.append("action", action);
  if (userId) params.append("user_id", userId);
  if (status) params.append("status", status);
  const { data } = await api.get(`/api/admin/logs?${params}`);
  return data;
}

export async function getModelInfo() {
  const { data } = await api.get("/api/admin/model");
  return data;
}

export async function switchModel(version) {
  const { data } = await api.patch("/api/admin/model", { version });
  return data;
}

export async function getAdminFeedback({ skip=0, limit=50 } = {}) {
  const { data } = await api.get(`/api/admin/feedback?skip=${skip}&limit=${limit}`);
  return data;
}