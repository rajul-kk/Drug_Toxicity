import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    outDir: 'web/static/dist',
    emptyOutDir: true,
    rollupOptions: {
      input: 'web/src/main.js',
      output: {
        entryFileNames: 'bundle.js',
        chunkFileNames: '[name].js',
        assetFileNames: '[name].[ext]',
      },
    },
  },
});
