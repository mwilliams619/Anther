# UI design overhaul — elevation

Status: **planned**, Phase 0 (preview page) built.
Scope: **`ui/static/style.css` only**, plus cache-bust bumps in
`ui/static/index.html`. No JS changes, no markup changes, no Python changes.

> **Two superseded directions**, both built and rejected on sight. Keeping them
> here because the reason they failed is the reason this one works.
>
> 1. **Glassmorphism on nodes + neumorphic plates in the sidebar.** Glass on a
>    node turns *data* into *ornament* — a rim light and a specular dot are
>    visual weight that encodes nothing. Neumorphism's grammar is **boxes**,
>    which is exactly what a dense tool sidebar does not want.
> 2. **"Studio at 2am" — warm room, songs as light sources casting pools.**
>    Too much. An atmospheric effect over the map competes with the map.
>
> The common failure: **both tried to make near-black emit light.** On a
> `#0A0A0A` surface there is no headroom above it for a highlight and nothing
> beneath it for a shadow to darken, so any attempt reads as an *effect* laid
> on top rather than as depth. Do not reintroduce either.

## The idea

No metaphor and no atmosphere. Depth is z-height and nothing else: things sit
at different levels, higher things are lighter and cast onto what's below, and
that is the entire system.

**Height is carried first by surface lightness, and only reinforced by
shadow.** This is the inversion that makes it work on a dark UI — a shadow
needs something lighter than itself to fall on, and near-black provides none,
which is why the previous two attempts had to resort to effects.

**The map gets nothing.** It is `z0`, the floor everything else rests on, and
being the darkest plane is its whole treatment. That is not a compromise: it
makes the map recede and the content stand out for free, and it means the graph
rendering — nodes, edges, grain, the existing hover and selection states — is
untouched.

## The ladder

| Level | What | Surface | Shadow |
|---|---|---|---|
| `z0` | the map | `#0D0D0D` flat | — receives |
| `z1` | sidebar, mentor panel | `#191919 → #141414` | boundary falloff onto the map |
| `z2` | detail panel, transport bar, filter suggest | `#222222 → #1A1A1A` | `--elev-2` |
| `z3` | settings popover, tooltip, modals, toast | `#282828 → #1F1F1F` | `--elev-3` |

```css
--z0:     #0D0D0D;
--z1-top: #191919;  --z1-bot: #141414;
--z2-top: #222222;  --z2-bot: #1A1A1A;
--z3-top: #282828;  --z3-bot: #1F1F1F;

/* Warm-biased black, so the dark never reads blue. You don't perceive it
   as warm — it just stops looking cold. This is the only warmth left from
   the "studio at 2am" draft, and it survives on its own merits. */
--elev-1: 0 1px 2px rgba(20,15,10,0.40), 0 3px 8px rgba(20,15,10,0.22);
--elev-2: 0 4px 10px rgba(20,15,10,0.45), 0 14px 30px rgba(20,15,10,0.40);
--elev-3: 0 10px 20px rgba(20,15,10,0.50), 0 28px 60px rgba(20,15,10,0.50);

--border: #2A2A2A;   /* was #4A4A4A */
```

Everything else — text greys, `--primary`, `--success`, `--warning`,
`--danger`, the 3px radius, Inter, the mono stack — is untouched.

## The three things that carry it

Named explicitly because they were identified as the parts that actually
worked, and because each is doing a distinct job:

1. **Cast shadow.** Makes an element sit *above* the map rather than beside it.
2. **Boundary falloff.** A soft dark gradient down the map's left edge where
   the sidebar meets it, replacing the hard 1px border. The panel ends because
   its shadow runs out, not because a line says so.
3. **Internal top-to-bottom gradient.** Each surface being slightly lighter at
   its top gives it a form of its own instead of being one flat fill. It is
   subtle to the point of being subliminal on a 300px panel and it is most of
   what separates this from a flat theme.

**Deliberately not used: a lit top edge.** A 1px highlight along the top of
each surface was offered and not chosen. The surface gradient already implies
the light direction, and a bright hairline on a dark UI reads as a drawn line
rather than as light. Do not add one back.

**Borders drop from `#4A4A4A` to `#2A2A2A`.** Elevation now does the
separating, so a bright hairline is just noise — and on a dark UI it is the
loudest thing on screen. This single change is a large share of the visible
improvement.

**Controls stay flat.** Buttons, chips, and rows get no elevation of their own.
Depth belongs to containers; giving it to every clickable thing inside them is
what made the neumorphic draft look quilted. The one exception is inputs, which
sit slightly *below* their panel — the only place a recess is honest, since you
are putting something into them.

---

## Phase 0 — preview page (done)

`ui/static/design-preview.html`. Self-contained, no build step, no server.
Open it directly or hit `/static/design-preview.html` while the app runs.

- **`D`** — A/B against the current design
- **`E`** — label every surface with its level and hex values

Judge specifically: whether four levels are distinguishable without the labels
on, whether the sidebar gradient is perceptible at all (it is the subtlest
piece and the easiest to lose), and whether dropping the borders to `#2A2A2A`
loses definition anywhere it was needed.

## Phase 1 — tokens

Add the ladder and the three `--elev-*` shadows to `:root`. Repoint `--border`.
Nothing else in `:root` changes. Every later phase is then a matter of pointing
existing rules at these tokens.

## Phase 2 — z1, the panels

`.panel-left` and `.panel-right` take the `z1` gradient and **lose their
`border-right` / `border-left`**. A `.boundary` element in `.graph-wrap`
provides the falloff. Section plates are not reintroduced — that was the
rejected draft.

## Phase 3 — z2, the cards

`#detail-panel`, `.autoplay-bar`, `#filter-suggest`: `z2` gradient,
`--elev-2`, borders removed. `#detail-panel` currently sits at
`rgba(26,26,26,0.98)` and becomes fully opaque, which is simpler and reads
better over a busy map.

## Phase 4 — z3, the top of the ladder

`.settings-popover`, `#graph-tooltip`, `.modal-content`, `#error-toast`,
`.mentor-toggle-btn`: `z3` gradient, `--elev-3`, borders removed.
`.modal-overlay` stays as it is.

## Phase 5 — the map, and the rest

The only map change is `--graph-bg` to `--z0` (`#101010` → `#0D0D0D`). Node
fills, strokes, edge opacities, grain, hover, selection, playback classes, and
the idle breathing are all untouched.

Two accessibility gaps to close while in here, both pre-existing:
`:focus-visible` rings on every control — only `.view-toggle-btn` has one
today — and `prefers-reduced-motion` on `.idle-breathing`.

Finally, `index.html` pins `style.css?v=12`. Bump it, or returning users get
cached old CSS. `graph.js?v=6` does not need bumping, because `graph.js` is not
being touched.

## Risks

There are unusually few, which is itself the argument for this direction.

| Risk | Response |
|---|---|
| Four levels are indistinguishable in practice | The preview's `E` key exists precisely to check this. If `z2` and `z3` collide, widen the gap rather than adding borders back |
| Dropping borders loses definition somewhere | Most likely on `#filter-suggest` and `.modal-content`, which overlay similarly-lit surfaces rather than the dark map. Check those two specifically |
| Lighter surfaces raise OLED power draw and look washed in bright rooms | The whole ladder spans `#0D0D0D` to `#282828`, still firmly dark. If it reads light, lower all four in step rather than compressing them |
| The map now looks plain next to designed chrome | Accepted deliberately: two attempts to treat the map were rejected. If it does read as plain, the next lever is legibility work — node sizing, label contrast, cluster readability — not atmosphere |

## Not doing

No glassmorphism, no neumorphism, no `backdrop-filter`, no light pools, no
haze, no vignette, no per-node ornament, no radius change, no typography
change, no light theme, no new motion, and no changes to `graph.js` or
`artist-graph.js`.
