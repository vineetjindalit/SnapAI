// frontend-react/src/api/types.ts
// Mirrors the JSON shapes the Python server emits. Keep these in sync with
// backend/api/pipeline.py and backend/api/routes.py.

export interface HealthResponse {
  status: string;
  version: string;
  sessions: number;
  auth_enabled?: boolean;
  dev_ui?: boolean;          // owner-only Developer toggle (SNAPPY_DEV_UI=1)
  persistence: boolean;
  compute: { device: string; torch_version?: string; mps_available?: boolean;
             cuda_available?: boolean; cuda_device_name?: string; cuda_mem_gb?: number };
  continuous_learning: boolean;
  models: {
    face:      { backend: string };
    gaze?:     { backend: string };
    clip:      { available: boolean; error: string | null;
                 trained_centroids?: boolean; trained_meta?: any };
    nima:      { available: boolean; error: string | null };
    hsemotion: { available: boolean; error: string | null };
    whisper?:  { available: boolean; error: string | null };
    calibration?: { available: boolean };
  };
}

export interface SessionResponse {
  session_id: string;
  event_name: string;
  event_type: string;
  prompt: string;
  ws_url: string;
  owner_id?: number;
}

// ── Admin (owner-only Mac dashboard) ────────────────────────────────────
export interface AdminUser {
  id: number;
  email: string;
  created: number;
  last_login: number | null;
  role: string;
  tier: string;
}

export interface AdminSession {
  sid: string;
  owner_id: number;
  owner_email: string | null;
  event_name: string;
  event_type: string;
  created: number;
  ended: number | null;
  active: boolean;
  captures: number;
  video_status: string | null;
  video_error: string | null;
  issues: string[];
}

export interface AdminOverviewResponse {
  users: AdminUser[];
  sessions: AdminSession[];
}

export interface MomentResult {
  detected_moment: string;
  confidence: number;
  all_scores: Record<string, number>;
  active_moments: string[];
  is_capture_moment: boolean;
}

export interface EmotionResult {
  dominant: string;
  score: number;
  per_emotion: Record<string, number>;
  happy_ratio: number;
  avg_smile: number;
}

export interface GazeResult {
  total_faces: number;
  faces_looking: number;
  gaze_ratio: number;
  is_group_looking: boolean;
  confidence: number;
  backend: string;
}

export interface ShotAnalysis {
  timestamp: number;
  blur: number; faces: number; face_q: number;
  comp: number; bright: number; emotion: number;
  total: number; nima?: number;
  // Legacy aliases the older index.html used
  blur_score?: number;
  composition_score?: number;
  emotion_score?: number;
  brightness_score?: number;
  face_quality?: number;
  total_score?: number;
}

export interface PredictionResult {
  trend: "rising" | "falling" | "peak" | "stable";
  peak_in_ms?: number;
}

export interface FaceBox { x: number; y: number; w: number; h: number; }

export interface CLIPInfo {
  available: boolean;
  backend: string;
  best: string | null;
  best_score: number | null;
  infer_ms: number | null;
  kept_centroid_sim: number | null;
  init_error: string | null;
}

export interface EnsembleDecision {
  triggered: boolean;
  final_score: number;
  threshold: number;
  moment_class: string;
  reasons: string[];
  contributions: Record<string, number>;
  weights: Record<string, number>;
  calibrated_p: number | null;
  priority_tier?: string;
  bypass_throttle?: boolean;
}

export interface FrameResult {
  type: "frame_result";
  frame_id: number;
  analysis: ShotAnalysis;
  face_detection: { available: boolean; backend: string;
                    face_count: number; boxes: FaceBox[]; error: string | null };
  gaze: GazeResult;
  moment: MomentResult;
  emotion: EmotionResult;
  prediction: PredictionResult;
  clip: CLIPInfo;
  nima: { available: boolean; score: number | null };
  hsemotion: { available: boolean };
  engine: EnsembleDecision | null;
  calibration?: { p: number } | null;
  // Setup coach (live only): placement guidance + readiness 0-100.
  setup?: {
    tips: string[];
    score: number;
    checklist?: Array<{ label: string; ok: boolean; detail: string }>;
    guide?: string[];
  } | null;
  captured: boolean;
  capture_reason: string | null;
  captured_url: string | null;
  captured_zoom_url: string | null;
  captured_zoom_info: { x: number; y: number; w: number; h: number;
                         zoom_factor: number;
                         url?: string; filepath?: string } | null;
  captured_thumb: string | null;
  total_captures: number;
  score_timeline: number[];
  gallery_frame: { id: number; b64: string; score: number; faces: number;
                   moment: string; mconf: number; gaze: boolean;
                   captured: boolean; ts: number };
  active_moments: string[];
  audio?: { events: Record<string, number>; boost: number };
  sequence?: { event: string; stage: number | null; cursor: number };
  source?: string;        // "upload" when streamed from a recorded clip
  media_ts?: number;      // media timestamp (s) of this frame, when uploading
  new_discoveries?: Array<any>;
}

export interface PromptTransition {
  ts: number;
  text: string;
  intent: "add" | "remove" | "replace" | "noop";
  matched: string[];
  active: string[];
  note: string;
}

/** Custom Mode's conversational turn — SnapAI's reply to a free-text/voice
 *  capture request ("capture me when I pose"), plus the live guide state. */
export interface AssistantReply {
  status: "idle" | "understanding" | "watching" | "captured";
  reply: string;
  watching_for: string;
  guide: string;
  /** General Mode only — the derived moment list as a live checklist (each
   *  moment's "captured" flips true the first time it fires). Absent for
   *  Custom Mode, which has a single objective, not a list. */
  moments?: Array<{ label: string; captured: boolean }>;
}
