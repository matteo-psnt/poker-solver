import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// Separate from `vite.config.ts` on purpose: vite's `defineConfig` does not
// accept a `test` key, and the merged type is more friction than a second file.
export default defineConfig({
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  // `testTimeout` must stay clear of the 5s `asyncUtilTimeout` in test-setup, or
  // vitest kills the test before `waitFor` can report which query never matched.
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
    testTimeout: 20_000,
  },
});
