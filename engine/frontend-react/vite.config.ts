import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { fileURLToPath, URL } from "node:url";
import { readFileSync, existsSync } from "node:fs";

// Field-test mode: `npm run dev:lan` sets SNAPPY_LAN=1, which binds the dev
// server to the whole LAN (so a phone on the same Wi-Fi can reach it) and
// serves HTTPS (phone browsers block the camera on a non-secure origin). The
// self-signed cert lives in .certs/ (generated once, gitignored). Normal
// `npm run dev` is unaffected — stays localhost / http.
const LAN = process.env.SNAPPY_LAN === "1";
const certDir = fileURLToPath(new URL("./.certs", import.meta.url));
const httpsOpt =
  LAN && existsSync(`${certDir}/snappy-cert.pem`)
    ? {
        key: readFileSync(`${certDir}/snappy-key.pem`),
        cert: readFileSync(`${certDir}/snappy-cert.pem`),
      }
    : undefined;

// During `npm run dev`, the dev server lives on :5173 and proxies API calls
// to the Snappy server on :8765 so cookies / WS / file uploads "just work".
//
// `npm run build` outputs static files to ./dist which the Snappy backend
// (backend/api/server.py) auto-serves at "/" if dist/index.html exists.
export default defineConfig({
  plugins: [
    react(),
    // Installable "app": adds a home-screen icon + standalone (no browser
    // chrome) launch + an offline-capable shell. No App Store review, no
    // separate codebase — this IS the existing site, just installable.
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["apple-touch-icon.png"],
      manifest: {
        name: "SnapAI — AI Event Photographer",
        short_name: "SnapAI",
        description: "AI photographer that auto-captures the best moments of your event.",
        theme_color: "#6366f1",
        background_color: "#0b0b12",
        display: "standalone",
        orientation: "portrait",
        start_url: "/",
        icons: [
          { src: "icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "icon-512.png", sizes: "512x512", type: "image/png" },
          { src: "icon-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
      workbox: {
        // Never cache API/WS calls or captured photos — only the app shell
        // (JS/CSS/HTML) should be served from cache. Live camera data must
        // always be fresh.
        navigateFallbackDenylist: [/^\/(sessions|auth|billing|captures|albums|health|ws|admin)\//],
        runtimeCaching: [
          {
            urlPattern: ({ url }) =>
              !/^\/(sessions|auth|billing|captures|albums|health|ws|admin)\//.test(url.pathname),
            handler: "NetworkFirst",
            options: { cacheName: "snappy-shell" },
          },
        ],
      },
    }),
  ],
  resolve: {
    // Mirror the `@/*` path alias from tsconfig.json. tsconfig is only
    // consulted by the TypeScript compiler for type-checking; Vite needs
    // its own resolve.alias entry to rewrite imports at bundle time.
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    host: LAN ? true : "localhost",   // LAN mode binds 0.0.0.0 for phones
    https: httpsOpt,                  // HTTPS so the phone camera is allowed
    proxy: {
      "/sessions":   { target: "http://localhost:8765", changeOrigin: true },
      "/auth":       { target: "http://localhost:8765", changeOrigin: true },
      "/billing":    { target: "http://localhost:8765", changeOrigin: true },
      "/captures":   { target: "http://localhost:8765", changeOrigin: true },
      "/albums":     { target: "http://localhost:8765", changeOrigin: true },
      "/health":     { target: "http://localhost:8765", changeOrigin: true },
      "/ws":         { target: "ws://localhost:8765", ws: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    chunkSizeWarningLimit: 1000,
  },
});
