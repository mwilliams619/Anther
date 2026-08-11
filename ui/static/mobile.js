/* ── Mobile shell — design 1b, "one surface, thumb height" ──────────────────
 *
 * Phones get the same app re-hosted in a bottom sheet instead of a left
 * drawer. A persistent bar at the bottom is the search field; drag it up and
 * it becomes the whole panel with tabs; tapping a node swaps that same sheet
 * to the detail view. Everything you touch lives in the lower third, and the
 * map keeps the full screen behind it.
 *
 * This is a re-host, not a fork. No markup is duplicated: the existing
 * elements (#view-mode, .panel-left, #detail-panel, #settings-popover,
 * #mentor-toggle) are *moved* into the sheet, so app.js keeps driving them by
 * id and every listener it attaches survives the move. Every move is recorded
 * so the shell can be torn down again when a rotation takes the viewport back
 * over the breakpoint.
 *
 * Nothing here runs on desktop — activation is gated on the same 768px
 * breakpoint the rest of the mobile CSS uses, and `body.mshell` is the switch
 * every mobile-shell rule in style.css hangs off.
 */
const AtlasMobile = (function () {
  const MQ = window.matchMedia('(max-width: 768px)');

  // Which panel sections belong to which tab. Song/artist pairs both appear:
  // setViewMode() hides the wrong one of each pair with [hidden], and the tab
  // only decides which *group* is in play.
  const TABS = {
    search:   ['song-search-section', 'artist-search-section'],
    map:      ['song-map-section', 'artist-map-section'],
    upload:   ['upload-section'],
    settings: ['mob-settings-tab'],
  };
  const TAB_LABELS = { search: 'Search', map: 'My map', upload: 'Upload', settings: 'Settings' };

  let active = false;
  let sheet, scrim, topBar, countEl, settingsTab;
  let tab = 'search';
  let moves = [];          // [{node, parent, next}] — for teardown
  let observers = [];
  let drag = null;

  /* ── helpers ───────────────────────────────────────────────────────────── */
  const $ = id => document.getElementById(id);

  function el(tag, cls, html) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }

  /* Move a node, remembering where it came from. */
  function move(node, parent) {
    if (!node) return null;
    moves.push({ node, parent: node.parentNode, next: node.nextSibling });
    parent.appendChild(node);
    return node;
  }

  function barHeight() {
    // The collapsed height, read off the same declarations the CSS uses so the
    // drag snap and the stylesheet can never disagree. The safe-area part is
    // taken from the sheet's resolved padding rather than from --mob-safe:
    // getPropertyValue hands back an unresolved `env(...)` for a custom
    // property, which parseFloat can only turn into NaN.
    const bar  = parseFloat(getComputedStyle(document.body).getPropertyValue('--mob-bar-h')) || 82;
    const safe = parseFloat(getComputedStyle(sheet).paddingBottom) || 0;
    return bar + safe;
  }

  /* ── build ─────────────────────────────────────────────────────────────── */
  function build() {
    const app = document.querySelector('.app');
    const panel = document.querySelector('.panel-left');

    /* Top chrome: brand + count on the left, map controls on the right. The
       header inside the panel is hidden on mobile — it would otherwise only be
       reachable by expanding the sheet. */
    topBar = el('div', 'mob-top');
    const brand = el('div', 'mob-brand',
      '<span class="mob-brand-name">Anther</span><span class="mob-count" id="mob-count"></span>');
    const right = el('div', 'mob-top-right');
    topBar.append(brand, right);

    move($('view-mode'), right);
    move($('mentor-toggle'), right);
    const gear = el('button', 'mob-icon-btn', '⚙');
    gear.type = 'button';
    gear.title = 'Settings';
    gear.setAttribute('aria-label', 'Settings');
    gear.addEventListener('click', () => { setTab('settings'); setState('expanded'); });
    right.appendChild(gear);

    /* Scrim — tapping it puts the sheet back down. */
    scrim = el('div', 'mob-scrim');
    scrim.addEventListener('click', () => collapse());

    /* The sheet itself. */
    sheet = el('div', 'mob-sheet');
    sheet.dataset.state = 'bar';

    const grip = el('div', 'mob-grip', '<span class="mob-handle"></span>');

    const bar = el('div', 'mob-bar');
    const field = el('button', 'mob-searchfield',
      '<span class="mob-searchfield-icon">⌕</span>Search songs, artists…');
    field.type = 'button';
    field.addEventListener('click', () => {
      setTab('search');
      setState('expanded');
      const artists = $('artist-search-section') && !$('artist-search-section').hidden;
      const input = $(artists ? 'artist-search-input' : 'search-input');
      // Focus after the height transition so the keyboard doesn't fight it.
      setTimeout(() => input && input.focus(), 260);
    });
    const tour = el('button', 'mob-tourbtn', '⇉');
    tour.type = 'button';
    tour.title = 'Play the map as a tour';
    tour.setAttribute('aria-label', 'Play the map as a tour');
    tour.addEventListener('click', startTour);
    bar.append(field, tour);

    const tabs = el('div', 'mob-tabs');
    tabs.setAttribute('role', 'tablist');
    Object.keys(TABS).forEach(name => {
      const b = el('button', 'mob-tab', TAB_LABELS[name]);
      b.type = 'button';
      b.dataset.tab = name;
      b.setAttribute('role', 'tab');
      b.addEventListener('click', () => setTab(name));
      tabs.appendChild(b);
    });

    sheet.append(grip, bar, tabs);

    /* The settings popover becomes an ordinary tab pane down here. */
    settingsTab = el('div', 'mob-settings-tab');
    settingsTab.id = 'mob-settings-tab';
    // Keep the reference: settingsTab is still detached here, so a second
    // getElementById for the popover would come back null.
    move($('settings-popover'), settingsTab).hidden = false;
    const help = el('button', 'btn mob-help-btn', 'Help &amp; tips');
    help.type = 'button';
    help.addEventListener('click', () => { collapse(); $('help-toggle').click(); });
    settingsTab.appendChild(help);
    panel.appendChild(settingsTab);
    moves.push({ node: settingsTab, parent: null, next: null });   // created here → removed on teardown

    move(panel, sheet);
    move($('detail-panel'), sheet);

    app.append(scrim, topBar, sheet);
    moves.push({ node: scrim, parent: null, next: null });
    moves.push({ node: topBar, parent: null, next: null });
    moves.push({ node: sheet, parent: null, next: null });

    countEl = $('mob-count');
    initDrag(grip);
    setTab('search');
    syncCount();
  }

  /* ── state ─────────────────────────────────────────────────────────────── */
  function setState(next) {
    if (!sheet) return;
    sheet.style.height = '';
    sheet.dataset.state = next;
    document.body.classList.toggle('mob-sheet-open', next !== 'bar');
  }

  function collapse() {
    if (!sheet) return;
    // Collapsing out of the detail view means "I'm done with this song", so
    // deselect rather than leaving a pinned node behind an invisible panel.
    if (sheet.dataset.state === 'detail') $('detail-close').click();
    else setState('bar');
  }

  function setTab(name) {
    if (!TABS[name]) return;
    tab = name;
    Object.entries(TABS).forEach(([key, ids]) => {
      ids.forEach(id => {
        const node = $(id);
        if (node) node.classList.toggle('mob-hidden', key !== name);
      });
    });
    sheet.querySelectorAll('.mob-tab').forEach(b =>
      b.classList.toggle('active', b.dataset.tab === name));
  }

  /* ⇉ in the bar: tour the map from the selected song, else from the first
     song the user added, else from whatever is on the map. */
  function startTour() {
    // Bare `AtlasGraph`, not `window.AtlasGraph` — graph.js declares it with
    // const, which lives in the global lexical scope and never on window.
    const nodes = AtlasGraph.getNodes();
    if (!nodes.length) {
      showError('Add some songs to the map first');
      return;
    }
    const selected = AtlasGraph.getSelectedId();
    const seed = selected
      || (nodes.find(n => n.kind === 'query') || nodes[0]).id;
    collapse();
    playFromHere(seed);
  }

  /* ── drag / tap on the grip ────────────────────────────────────────────── */
  function initDrag(grip) {
    grip.addEventListener('pointerdown', e => {
      grip.setPointerCapture(e.pointerId);
      drag = { y: e.clientY, h: sheet.getBoundingClientRect().height, moved: 0 };
      sheet.classList.add('dragging');
    });

    grip.addEventListener('pointermove', e => {
      if (!drag) return;
      const dy = e.clientY - drag.y;
      drag.moved = Math.max(drag.moved, Math.abs(dy));
      const h = Math.min(window.innerHeight * 0.9,
                         Math.max(barHeight(), drag.h - dy));
      sheet.style.height = h + 'px';
    });

    const end = () => {
      if (!drag) return;
      const h = sheet.getBoundingClientRect().height;
      const wasDetail = sheet.dataset.state === 'detail';
      const tapped = drag.moved < 6;
      drag = null;
      sheet.classList.remove('dragging');
      sheet.style.height = '';

      if (tapped) {
        if (wasDetail) collapse();
        else setState(sheet.dataset.state === 'expanded' ? 'bar' : 'expanded');
        return;
      }
      if (wasDetail) {
        // Only a decisive pull-down dismisses the detail; anything else snaps
        // back, so a stray drag never loses the song you were reading.
        if (h < window.innerHeight * 0.42) collapse(); else setState('detail');
        return;
      }
      setState(h > (barHeight() + window.innerHeight * 0.72) / 2 ? 'expanded' : 'bar');
    };
    grip.addEventListener('pointerup', end);
    grip.addEventListener('pointercancel', end);
  }

  /* ── wiring to app.js state ────────────────────────────────────────────── */
  function observe(node, fn, opts) {
    if (!node) return;
    const o = new MutationObserver(fn);
    o.observe(node, opts);
    observers.push(o);
  }

  function syncCount() {
    if (!countEl) return;
    const artists = $('artist-map-section') && !$('artist-map-section').hidden;
    const src = artists ? $('artist-map-count') : $('map-count');
    // "12 songs · 340 nodes" → "12 songs"; the pill has room for one fact.
    countEl.textContent = src ? (src.textContent || '').split('·')[0].trim() : '';
  }

  /* Per-activation: observers, torn down again with the shell. */
  function wire() {
    const detail = $('detail-panel');

    // The sheet follows the detail panel rather than intercepting selection,
    // so every path that opens one (node tap, neighbour row, tour) lands here.
    observe(detail, () => {
      // A touch can leave a synthetic hover tooltip visible for one frame;
      // remove it whenever the sheet becomes the detail surface.
      const tip = $('graph-tooltip');
      if (tip) tip.style.display = 'none';
      if (!detail.hidden) setState('detail');
      else if (sheet.dataset.state === 'detail') setState('bar');
    }, { attributes: true, attributeFilter: ['hidden'] });

    observe($('map-count'), syncCount, { childList: true, characterData: true, subtree: true });
    observe($('artist-map-count'), syncCount, { childList: true, characterData: true, subtree: true });
  }

  /* Once per page: delegated listeners on elements that outlive the shell, so
   * repeated rotations can't stack duplicates. Both no-op while inactive. */
  function wireOnce() {
    // Switching song/artist atlas re-hides sections; re-apply the active tab
    // on top of that and refresh the pill.
    $('view-mode').addEventListener('click', () => setTimeout(() => {
      if (!active) return;
      setTab(tab);
      syncCount();
    }, 0));

    // Acting on a row means you want to see the result on the map, so get the
    // sheet out of the way. Inputs and mode switches keep it up.
    document.querySelector('.panel-left').addEventListener('click', e => {
      if (!active) return;
      if (e.target.closest('input, label, .mode-btn, .view-toggle-btn, .settings-popover')) return;
      if (e.target.closest('.track-row, .map-row, .similar-row, .artist-row, #filter-suggest')) {
        setTimeout(() => { if (sheet.dataset.state === 'expanded') setState('bar'); }, 120);
      }
    });
  }

  /* ── activate / deactivate ─────────────────────────────────────────────── */
  function activate() {
    if (active) return;
    active = true;
    document.body.classList.add('mshell');
    // The old left-drawer shell and this one can't both be running.
    document.querySelector('.app').classList.remove('sidebar-open');
    build();
    wire();
  }

  function deactivate() {
    if (!active) return;
    active = false;
    observers.forEach(o => o.disconnect());
    observers = [];
    // Restore in reverse so each node goes back to a parent that still exists.
    moves.reverse().forEach(({ node, parent, next }) => {
      if (parent) parent.insertBefore(node, next);
      else node.remove();
    });
    moves = [];
    $('settings-popover').hidden = true;
    document.body.classList.remove('mshell', 'mob-sheet-open');
    sheet = scrim = topBar = countEl = settingsTab = null;
  }

  function sync() { if (MQ.matches) activate(); else deactivate(); }

  wireOnce();
  sync();
  // Safari < 14 only has the deprecated listener form.
  if (MQ.addEventListener) MQ.addEventListener('change', sync);
  else MQ.addListener(sync);

  return {
    isActive: () => active,
    expand: () => setState('expanded'),
    collapse,
    setTab,
  };
})();
