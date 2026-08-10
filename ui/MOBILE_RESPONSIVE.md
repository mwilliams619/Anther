# Mobile — moved

The phone UI is no longer the left drawer this file used to describe. It is a
bottom sheet (`ui/static/mobile.js` + the `body.mshell` block at the end of
`ui/static/style.css`), built from option **1b** of `ui/Anther Mobile.dc.html`.

Documentation lives with the rest of the UI:
[docs/ui.md § Mobile shell](../docs/ui.md#mobile-shell--one-surface-at-thumb-height).

The old drawer (`.sidebar-toggle` / `.app.sidebar-open`) is still in `app.js`
and `style.css` and still stands down automatically wherever the sheet
activates — it is the fallback for the 768px band a rotation can land in.
