# Full-Song Playback — Attempted and Reverted

**Status: tried in July 2026, measured, reverted. Not implemented. Reference only.**

Playback today is unchanged and stays as it was: a **30s Deezer preview** on the
▶ button (`atlas.get_preview_url`) and the **Spotify iframe** on the 🎵 button
(`atlas.get_spotify_track_id`), which gives a full track to visitors already
logged into Spotify and a 30s preview to everyone else.

---

## What was attempted

Replace the 30s previews with full songs from a no-login source. The build was:
`yt-dlp` `ytsearch1:` to resolve a YouTube video id per track (metadata only,
never stream extraction), a SQLite resolution cache, a `/api/song/<id>/stream`
endpoint returning a source ladder, and a shared player card rendering the
YouTube IFrame player with silent fallback to Spotify then Deezer.

## Why it was reverted

**1. Only ~33% of tracks actually played.** Audited over corpus tracks through
the real UI: 4/12 reached full-song YouTube playback. The rest failed with
YouTube **error 150** — the video refuses to embed. Three explanations were
ruled out: it was not the `playerVars.origin` param (same failures with and
without), not the headless test browser (other videos played fine in the same
session), and **not predictable at resolve time** — `yt-dlp` reports
`playable_in_embed: true` for every blocked video, so candidates cannot be
pre-filtered. Blocked videos skewed to major-label uploads, which is the bulk of
an MPD-derived corpus.

**2. The video panel damaged the atlas.** YouTube's embedded player has a
200x200 minimum viewport, and the API Developer Policies forbid separating audio
from video (III.I.7) and playing from a hidden/background player (III.I.9). So a
visible video panel is mandatory — it cannot be reduced to an audio strip. It
competed for attention with the graph, which is the actual product.

Neither problem was tunable. Both follow from the embed being the only legal
browser path.

## Why Nuclear can do this and we cannot

[Nuclear](https://nuclearplayer.com/) plays full songs, audio-only, from
YouTube. It is worth being precise about why that does not transfer, because the
difference is architectural, not a matter of effort.

Nuclear never uses YouTube's player. Its `api.Ytdlp.getStream(videoId)` returns
a **direct audio stream URL** (`mimeType: 'audio/webm'`, `codec: 'opus'` — an
audio-only itag), extracted by a `yt-dlp` binary on the user's own machine and
played in a native player. The IFrame API, the 200x200 minimum, and the
no-background-player policy are simply not in its path. Audio-only is not
something Nuclear negotiated; it is what you get when you skip the embed.

That route is closed to us for three independent reasons:

- **Attribution.** Nuclear is desktop software: extraction traffic comes from
  each user's own machine. Anther is one branded, identifiable host serving an
  audience that includes label A&R.
- **The browser cannot use those URLs anyway.** `googlevideo` URLs are bound to
  the IP that resolved them and are CORS-blocked, so every stream would have to
  be proxied through Flask — putting the audio bytes on the host box.
- **Delivery.** The tunnel in front of the app restricts disproportionate
  non-HTML content; audio streaming is the named case.

Nuclear's robustness *is* raw stream extraction. Take that away and what remains
is embeds — which is what was measured above.

Also relevant: Nuclear's MCP server is a remote control (domains `Queue`,
`Playback`, `Metadata`, `Favorites`, `Playlists`, `Dashboard`, `Providers` —
`Streaming` is **not** exposed), it is localhost-only, and its `call` bridge
requires a live webview, so it cannot run headless. It cannot deliver audio to a
visitor's browser under any configuration.

## What was kept

`atlas.track_name_artist()` — extracted while building the resolver, it
de-duplicates the (name, artist) lookup that `get_preview_url` and
`get_spotify_track_id` both carried. It stands on its own and both still use it.

Everything else was removed: `ui/resolve.py`, `ui/resolve_cache.py`,
`ui/static/player.js`, the `/api/song/<id>/stream`, `/api/stream/batch` and
`/api/stream/status` routes, the `yt-dlp` dependency, and the player card CSS.

## If this is revisited

Re-measure before rebuilding. The embed rate was measured against
`http://127.0.0.1`; YouTube syndication rules are origin- and region-sensitive,
so the deployed domain may differ. That single number decides whether any of
this is worth repeating.

The only other browser-legal audio-only option is SoundCloud's widget with
`visual=false` (a ~20px strip, no login, full songs). It was not tried: coverage
skews indie, and on a mainstream MPD-derived corpus it would likely land below
YouTube's 33%.

Do **not** revisit proxying audio through Flask. See the three reasons above.

And regardless of playback source: full-length audio must never reach the
embedder. The corpus was built from Deezer 30s previews through
`clip_waveform()`, and `docs/invariants.md` requires query and corpus share one
transform.
