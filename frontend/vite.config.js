import { defineConfig } from 'vite'

export default defineConfig(({ mode }) => ({
  base: mode === 'production' ? '/privacy-face-recognition/' : '/',
  server: {
    port: 5173
  }
}))
