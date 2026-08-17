import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Your components fetch relative paths like "/api/patients" -- that's
// correct against a real deployment (frontend and backend behind the
// same origin/reverse proxy), but Vite's dev server runs on its own
// port (5173) separate from uvicorn's (8000). This proxy makes the dev
// server forward anything under /api to the FastAPI backend, so the
// fetch calls in api/*.js don't need to change between dev and prod.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
