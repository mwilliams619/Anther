# Mobile-Responsive UI Changes

## Summary

The Anther app is now fully responsive on mobile and tablet devices. Instead of being crushed by the left panel, the sidebar becomes a collapsible drawer that slides in from the left when you tap the hamburger menu.

## What Changed

### 1. **CSS** (`ui/static/style.css`)
Added a mobile media query (`@media (max-width: 768px)`) that:
- Hides the left panel off-screen by default (`transform: translateX(-100%)`)
- Makes it slide in when the `.sidebar-open` class is added to `.app`
- Shows a hamburger menu button (☰) in the top-left corner
- Adds a semi-transparent backdrop that closes the sidebar when clicked
- Full-width map is now visible on mobile

### 2. **HTML** (`ui/static/index.html`)
Added a single toggle button:
```html
<button class="sidebar-toggle" aria-label="Toggle sidebar">☰</button>
```
This button is hidden on desktop (via CSS) and appears only on screens ≤768px wide.

### 3. **JavaScript** (`ui/static/app.js`)
Added `initMobileSidebar()` function that:
- Toggles the sidebar when the hamburger button is clicked
- Closes the sidebar when clicking the map
- Closes the sidebar when selecting a search result or playlist
- Closes the sidebar when clicking the dark backdrop
- Preserves all functionality—just reorganizes the layout

Called from `init()` so it runs automatically on page load.

## Behavior

### Desktop (>768px)
- **No changes.** The layout remains the 2-column grid (sidebar + map) as before.
- The hamburger button exists in the DOM but is not visible.

### Mobile & Tablet (≤768px)
- **Map is always visible first.** The sidebar is hidden off-screen.
- **Tap the ☰ button** (top-left corner) to slide the sidebar in.
- **Tap on the map** to auto-close the sidebar (for quick interactions).
- **Tap a search result, playlist, or recommendation** to place it and auto-close.
- **Tap the dark backdrop** to close the sidebar without selecting anything.
- The sidebar toggle button moves to the right when the sidebar is open (better UX).

## Responsive Breakpoint

The breakpoint is set to **768px** (typical tablet threshold). You can adjust it:
```css
@media (max-width: 768px) {  /* Change 768px to another value if needed */
```

Suggested alternatives:
- `640px` — iPhone size only
- `1024px` — iPad or larger devices

## Testing

To test mobile view in Chrome/Firefox:
1. Open DevTools (F12 or Cmd+Opt+I)
2. Click the device-toggle icon (top-left of DevTools)
3. Select an iPhone or iPad preset, or manually set width to ≤768px
4. Reload the page — the hamburger button appears

## No Breaking Changes

- All existing functionality preserved
- Desktop experience identical
- Sidebar scroll behavior unchanged
- All API routes and graph interactions work as before
