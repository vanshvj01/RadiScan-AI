import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: 'class',
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        risk: {
          low: '#16a34a',
          moderate: '#ca8a04',
          high: '#ea580c',
          critical: '#dc2626',
        },
      },
    },
  },
  plugins: [],
};
export default config;
