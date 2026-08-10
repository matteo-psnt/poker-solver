import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
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
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        /**
         * Answer a dead backend as a dead backend.
         *
         * Vite's default is to leave the response to the next middleware, which
         * is the SPA fallback — so a refused connection came back as
         * `index.html` at status 200. The console then parsed the console as a
         * payload, and every panel reported a problem with its own data while
         * the real fact was that `serve` was not listening.
         *
         * It is not a rare state either: this recipe starts Vite and the Python
         * server together, and Vite is ready several seconds first. So the
         * FIRST load of every page hits it.
         *
         * 502 rather than 503: the console retries a 502 (`isTransient`) and
         * shows a 503 as a settled refusal from the cloud, which this is not.
         */
        configure: (proxy) => {
          proxy.on("error", (error, _request, response) => {
            if (!("writeHead" in response) || response.headersSent) return;
            response.writeHead(502, { "content-type": "application/json" });
            response.end(
              JSON.stringify({
                // The address comes from the proxy's own error, not from a
                // constant here — a hardcoded port would keep naming 8765
                // after someone pointed the target elsewhere.
                error: `The API server did not answer (${error.message}). It is probably still starting — \`console-dev\` brings it up alongside Vite.`,
              }),
            );
          });
        },
      },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
