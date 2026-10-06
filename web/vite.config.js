import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// /api goes to the FastAPI server started with uvicorn on port 8000.
const apiTarget = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { "/api": apiTarget },
  },
  preview: {
    port: 4173,
    proxy: { "/api": apiTarget },
  },
});
