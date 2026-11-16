import { defineConfig } from "vite";

// Tauri expects a fixed port and must see Rust errors in the terminal.
// fs.allow: the browser preview may show a frame from a workspace in the checkout.
export default defineConfig({
  clearScreen: false,
  server: { port: 1420, strictPort: true, watch: { ignored: ["**/src-tauri/**"] }, fs: { allow: [".."] } },
  build: { target: "es2022" },
});
