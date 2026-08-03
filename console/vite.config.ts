import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // `@/` is declared in tsconfig for the editor and the type-checker; Rollup
  // resolves independently and needs telling separately, or the build fails on
  // imports that type-check perfectly.
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    // Dev runs against the REAL server: there is no mock layer, by decision.
    // `just console-dev` starts both; this is the seam between them.
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
