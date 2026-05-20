import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: BASE_URL });


// ── Detection ──────────────────────────────────────────────────────────────

/**
 * Run damage detection on uploaded images.
 * @param {File[]} files         - array of image files (max 10)
 * @param {string} sessionName   - user-defined session name
 */
export async function detectDamage(files, sessionName) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await api.post(
    `/api/detect/?session_name=${encodeURIComponent(sessionName)}`,
    formData,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return response.data;
}


// ── Sessions ───────────────────────────────────────────────────────────────

/**
 * List sessions with optional search / filter / sort / pagination.
 */
export async function getSessions({
  skip     = 0,
  limit    = 20,
  search   = "",
  severity = "",
  sort     = "date_desc",
} = {}) {
  const params = new URLSearchParams({ skip, limit, sort });
  if (search)   params.append("search",   search);
  if (severity) params.append("severity", severity);

  const response = await api.get(`/api/sessions/?${params}`);
  return response.data;
}

/**
 * Get a single session with all its images.
 */
export async function getSession(sessionId) {
  const response = await api.get(`/api/sessions/${sessionId}`);
  return response.data;
}

/**
 * Rename a session.
 */
export async function renameSession(sessionId, name) {
  const response = await api.patch(`/api/sessions/${sessionId}/rename`, { name });
  return response.data;
}

/**
 * Delete an entire session.
 */
export async function deleteSession(sessionId) {
  const response = await api.delete(`/api/sessions/${sessionId}`);
  return response.data;
}

/**
 * Delete selected images from a session.
 * @param {number}   sessionId     - session ID
 * @param {number[]} detectionIds  - list of detection IDs to delete
 */
export async function deleteSessionImages(sessionId, detectionIds) {
  const response = await api.delete(`/api/sessions/${sessionId}/images`, {
    data: { detection_ids: detectionIds },
  });
  return response.data;
}

/**
 * Update an image's data after editing.
 * @param {number} sessionId     - session ID
 * @param {number} detectionId   - detection ID
 * @param {string} imageData     - base64 encoded edited image
 */
export async function updateDetectionImage(sessionId, detectionId, imageData) {
  const response = await api.patch(
    `/api/sessions/${sessionId}/images/${detectionId}`,
    { image_data: imageData }
  );
  return response.data;
}


// ── Legacy (keep for backwards compat) ────────────────────────────────────

/**
 * @deprecated use getSessions() instead
 */
export async function getHistory(skip = 0, limit = 20) {
  return getSessions({ skip, limit });
}


/**
 * Re-run YOLO detection on an already-saved edited image.
 */
export async function reDetectImage(sessionId, detectionId) {
  const response = await api.post(
    `/api/sessions/${sessionId}/images/${detectionId}/redetect`
  );
  return response.data;
}


/**
 * Save edited image as a new copy detection in the same session.
 */
export async function copyDetectionImage(sessionId, detectionId, imageData, copyName) {
  const response = await api.post(
    `/api/sessions/${sessionId}/images/${detectionId}/copy`,
    { image_data: imageData, copy_name: copyName }
  );
  return response.data;
}


/**
 * Get the URL for a raw (unannotated) image served from disk.
 * Use directly as <img src={getRawImageUrl(...)} />
 */
export function getRawImageUrl(sessionId, detectionId) {
  const base = import.meta.env.VITE_API_URL || "http://localhost:8000";
  return `${base}/api/sessions/${sessionId}/images/${detectionId}/raw`;
}