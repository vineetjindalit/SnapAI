/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // User mode — soft, warm, inviting
        cream:    { 50: "#fefdfb", 100: "#fdf6ec", 200: "#f7ead0" },
        rose:     { 500: "#f43f5e", 600: "#e11d48" },
        // Brand
        brand:    { 500: "#6366f1", 600: "#4f46e5", 700: "#4338ca" },
        // Dev mode — cool slate
        ink:      { 900: "#0f172a", 800: "#1e293b", 700: "#334155" },
      },
      fontFamily: {
        sans: ['ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont',
               '"Helvetica Neue"', 'Arial', 'sans-serif'],
        mono: ['ui-monospace', '"SF Mono"', 'Menlo', 'Consolas', 'monospace'],
      },
      animation: {
        "pulse-slow":   "pulse 3s ease-in-out infinite",
        "capture-flash":"flash 0.6s ease-out",
        "slide-in-up":  "slideInUp 0.4s ease-out",
      },
      keyframes: {
        flash: {
          "0%":   { opacity: "0", transform: "scale(0.95)" },
          "30%":  { opacity: "1" },
          "100%": { opacity: "0", transform: "scale(1.05)" },
        },
        slideInUp: {
          "0%":   { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};
