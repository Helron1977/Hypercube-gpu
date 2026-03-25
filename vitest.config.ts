import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        globals: true,
        environment: 'node',
        setupFiles: ['./tests/vitest.setup.ts'],
        exclude: ['**/node_modules/**', '**/dist/**', '**/tests/renders/**'],
    },
});
