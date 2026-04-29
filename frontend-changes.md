# Frontend Changes

## Dark/Light Mode Toggle Button

### What was added
A theme toggle button positioned fixed in the top-right corner of the UI, allowing users to switch between dark mode (default) and light mode.

### Files changed

**`frontend/index.html`**
- Added `#themeToggle` button element before the closing `</body>` tag
- Button contains inline SVG icons: sun (shown in dark mode) and moon (shown in light mode)
- Has `aria-label` for accessibility; label is updated dynamically by JS
- Bumped stylesheet/script cache-bust version to `v=10`

**`frontend/style.css`**
- Added `[data-theme="light"]` CSS variable overrides for a complete light palette (background, surface, text, border colors)
- Added `transition` rules on key structural elements (`body`, `.sidebar`, `.chat-container`, `.message-content`, inputs, buttons) for smooth 0.3s color transitions when toggling
- Added `.theme-toggle` button styles: fixed position top-right, 40px circle, border + surface background matching the existing aesthetic, hover/focus-visible states with the existing `--focus-ring` and `--primary-color`
- Icon visibility controlled via CSS: `.icon-sun` visible by default (dark mode), `.icon-moon` visible under `[data-theme="light"]`
- Added `.toggling` class that applies a brief `rotate(30deg)` transform for an animation on click

**`frontend/script.js`**
- Added an IIFE `initTheme()` that runs before DOMContentLoaded to read `localStorage` (falling back to `prefers-color-scheme`) and set `data-theme` on `<html>` — prevents flash of wrong theme on load
- Added `toggleTheme()`: flips `data-theme` attribute on `document.documentElement`, persists choice to `localStorage`, briefly adds/removes `.toggling` class for the spin animation
- Added `syncThemeToggleLabel()`: keeps `aria-label` in sync with current mode ("Switch to dark/light mode")
- Wired toggle button click listener in `setupEventListeners()`

---

## Light Theme Accessibility & Color Audit

### What was changed
Hardened the `[data-theme="light"]` block to fix four WCAG failures and add element-specific overrides for components that used hardcoded dark-mode colors.

### WCAG issues fixed

| Element | Problem | Fix |
|---|---|---|
| `--text-secondary` | `#64748b` barely passed AA (~4.6:1) | Changed to `#475569` (Slate 600, ~6.7:1) |
| `--border-color` | `#e2e8f0` failed 1.4.11 non-text contrast (<3:1) | Changed to `#94a3b8` (Slate 400, ~3.1:1) |
| `.error-message` color | `#f87171` failed AA (~2.4:1 on white) | Changed to `#b91c1c` (Red 700, ~7.4:1) |
| `.success-message` color | `#4ade80` failed AA (~1.9:1 on white) | Changed to `#15803d` (Green 700, ~6.0:1) |

### Files changed

**`frontend/style.css`**
- Expanded `[data-theme="light"]` variable block: added `--primary-color: #1d4ed8` (Blue 700, 6.1:1 on bg), `--primary-hover: #1e40af`, `--user-message`, `--focus-ring`, and corrected `--text-secondary` and `--border-color` (see table above)
- Added `[data-theme="light"] .message-content code/pre` overrides: replaces `rgba(0,0,0,0.2)` dark overlay with `#e2e8f0` neutral tint
- Added `[data-theme="light"] .error-message` and `.success-message` overrides with accessible dark-on-light text colors
- Added `[data-theme="light"]` welcome message shadow override: softened to `rgba(0,0,0,0.06)`
- Added `[data-theme="light"]` scrollbar overrides: track `#f1f5f9`, thumb `#94a3b8`, hover `#64748b`

**`frontend/index.html`**
- Bumped stylesheet cache-bust version to `v=11`
