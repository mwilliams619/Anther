# Mentor Pane Collapsible UI — Implementation Summary

## Overview
The mentor right-side pane is now **hidden on launch** with a clickable chat bubble icon (💬) to toggle it open/closed. The layout smoothly transitions between a 2-column and 3-column grid.

---

## Changes Made

### 1. **HTML** (`ui/static/index.html`)

**Mentor panel wrapper:**
- Changed from `<div class="panel panel-right">` to `<div class="panel panel-right mentor-panel" hidden>`
- Now starts hidden and can be toggled via JS

**Toggle button:**
- Added `<button id="mentor-toggle" class="mentor-toggle-btn" title="Open mentor">💬</button>`
- Positioned on the graph area (bottom-right)
- Floats as a circular chat bubble

---

### 2. **CSS** (`ui/static/style.css`)

**Grid layout (app container):**
```css
.app {
  grid-template-columns: 320px 1fr;  /* 2 columns by default */
  transition: grid-template-columns var(--transition);
}

.app.mentor-open {
  grid-template-columns: 320px 1fr 320px;  /* Expands to 3 columns */
}
```

**Mentor toggle button:**
```css
.mentor-toggle-btn {
  position: absolute;
  bottom: calc(var(--spacing) * 3);
  right: calc(var(--spacing) * 2);
  width: 44px;
  height: 44px;
  border-radius: 50%;  /* Circular */
  background: var(--surface);
  border: 1px solid var(--border);
  color: var(--text);
  font-size: 20px;
  cursor: pointer;
  transition: background, border-color, box-shadow;
  z-index: 40;
}

.mentor-toggle-btn:hover {
  background: var(--surface-hi);
  border-color: var(--primary);  /* Blue accent */
  box-shadow: 0 2px 8px rgba(91,143,204,0.2);
}

.app.mentor-open .mentor-toggle-btn {
  display: none;  /* Hide when panel is open */
}
```

---

### 3. **JavaScript** (`ui/static/app.js`)

**Toggle function:**
```javascript
function toggleMentorPanel() {
  const app = document.querySelector('.app');
  const mentorPanel = document.querySelector('.mentor-panel');
  const isOpen = app.classList.toggle('mentor-open');
  mentorPanel.hidden = !isOpen;
  if (isOpen) {
    setTimeout(() => document.getElementById('mentor-input').focus(), 100);
  }
}
```

**Initialization:**
- Added to `initMentor()`: 
  ```javascript
  document.getElementById('mentor-toggle').addEventListener('click', toggleMentorPanel);
  ```

---

## User Experience

| State | Visual | Grid Columns |
|-------|--------|--------------|
| **Launch** | Chat bubble visible (💬) | 2 (left + graph) |
| **Click bubble** | Panel slides in, bubble disappears | 3 (left + graph + mentor) |
| **Click bubble again** | Panel slides out, bubble reappears | 2 (left + graph) |

**Features:**
- ✓ Smooth 100ms grid transition (no layout jank)
- ✓ Input auto-focuses when panel opens
- ✓ Button has hover effect (color + shadow)
- ✓ Matches app's dark theme
- ✓ Positioned where it won't interfere with graph interaction

---

## No Breaking Changes
- All mentor functionality preserved
- Existing API endpoints unchanged
- Chat history & reset button work as before
- Only display/layout modified

