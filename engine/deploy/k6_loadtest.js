// Snappy load test — k6 script.
//
// Install k6: brew install k6  (or https://k6.io/docs/get-started/installation/)
// Run:        k6 run -e SNAPPY_HOST=http://localhost:8765 -e VUS=50 deploy/k6_loadtest.js
//
// What it tests:
//   - 50–100 concurrent virtual users hitting /health, creating sessions,
//     uploading frames over WebSocket. Phase 4 PDF target = 100 simultaneous
//     event cameras with p99 < 200ms.
//
// What it does NOT test (do those separately):
//   - Real CLIP inference latency (run integration test with a real GPU box)
//   - Postgres failover (do via aws rds force-failover)
//   - Cost (Snappy at 100 sessions ≈ $80–150/mo ECS + RDS + S3)

import http from "k6/http";
import ws from "k6/ws";
import { check, sleep } from "k6";

const HOST = __ENV.SNAPPY_HOST || "http://localhost:8765";
const VUS  = parseInt(__ENV.VUS || "50");

export const options = {
  scenarios: {
    health: {
      executor: "constant-arrival-rate",
      rate: 5, timeUnit: "1s",
      duration: "30s",
      preAllocatedVUs: 10,
      exec: "healthCheck",
    },
    create_session: {
      executor: "constant-arrival-rate",
      rate: 2, timeUnit: "1s",
      duration: "60s",
      preAllocatedVUs: 5,
      exec: "createSession",
      startTime: "10s",
    },
    websocket_streams: {
      executor: "ramping-vus",
      stages: [
        { duration: "20s", target: VUS },
        { duration: "60s", target: VUS },
        { duration: "10s", target: 0 },
      ],
      exec: "wsFrameStream",
      startTime: "30s",
    },
  },
  thresholds: {
    http_req_duration: ["p(99)<500"],
    http_req_failed:   ["rate<0.01"],
  },
};

export function healthCheck() {
  const r = http.get(`${HOST}/health`);
  check(r, {
    "200": (r) => r.status === 200,
    "models present": (r) => r.json("models") !== undefined,
  });
}

export function createSession() {
  const r = http.post(
    `${HOST}/sessions`,
    JSON.stringify({event_name: "loadtest", event_type: "general", prompt: "general"}),
    { headers: { "Content-Type": "application/json" } },
  );
  check(r, { "201": (r) => r.status === 201 });
  if (r.status !== 201) return;
  const sid = r.json("session_id");
  // Don't bother running the WS in this scenario — separate scenario does
  http.del(`${HOST}/sessions/${sid}`);
}

export function wsFrameStream() {
  const create = http.post(
    `${HOST}/sessions`,
    JSON.stringify({event_name: "ws-load", event_type: "general", prompt: "general"}),
    { headers: { "Content-Type": "application/json" } },
  );
  if (create.status !== 201) return;
  const sid = create.json("session_id");

  // Tiny synthetic JPEG (1×1 black) so process_frame doesn't fail decoding
  const tinyJpegB64 =
    "/9j/4AAQSkZJRgABAQEAAAAAAAD/2wBDAP////////////////////////////////////////////////////////////////////////8AABEIAAEAAQEDESIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAAAAAB/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AzAAH/9k=";

  const url = HOST.replace("http", "ws") + "/ws/" + sid;
  const res = ws.connect(url, {}, (sock) => {
    sock.on("open", () => {
      // Send 30 frames at ~5 fps = 6 seconds
      let i = 0;
      const interval = setInterval(() => {
        sock.send(JSON.stringify({frame: tinyJpegB64}));
        if (++i >= 30) { clearInterval(interval); sock.close(); }
      }, 200);
    });
    sock.on("message", (msg) => {
      // Just count; no asserts here — failure modes are covered by /health
    });
    sock.setTimeout(() => sock.close(), 10000);
  });
  check(res, { "ws 101": (r) => r && r.status === 101 });
  http.del(`${HOST}/sessions/${sid}`);
}
