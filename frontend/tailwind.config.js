/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        nature: {
          bg: '#0f1713',        // Deep forest shadow background
          card: '#16231d',      // Rich moss-tinted card surface
          border: '#22382e',    // Subtle soft green border
          sage: '#8fa89b',      // Muted natural secondary text
          turf: '#34d399',      // Fresh minty turf green accent
          warm: '#fbbf24',      // Soft warm amber highlight
        }
      },
      fontFamily: {
        display: ['Outfit', 'sans-serif'],
        sans: ['"Plus Jakarta Sans"', 'sans-serif'],
      },
    },
  },
  plugins: [],
}