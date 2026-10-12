# Vega1-seg

Ask where something is in a video ("where is the dog?") and get a mask of it on
every frame, computed entirely in the browser. Nothing is uploaded.

- Samples open with masks already made for their example queries.
- Typed queries load the image encoder, mask decoder and text encoder once
  (cached by the browser), examine 16 keyframes, then answer each new query
  with the small decoder only.
- Vega1 (the yes/no model) checks each frame for the object and the mask shows
  only where it agrees, which removes most masks of objects that are not there.
- Every query is a mask layer you can highlight or hide; each mask splits into
  objects that a click on the video highlights or un-highlights, followed from
  frame to frame.
- Views: overlay, box, matte, cutout. Exports: a matte video (WebM) for track
  mattes in a video editor, per-frame boxes (JSON), the current frame (PNG).
  Exports contain only what is highlighted and examine every frame.

## Files

| file | what |
|---|---|
| `index.html`, `seg.css` | the page |
| `app.js` | page logic: samples, queries, layers, views, export |
| `objects.js` | objects within masks: regions, frame-to-frame links, choices |
| `segment.js` | browser API (`Vega1Seg`): workers, keyframes, masks, tracking |
| `seg-worker.js` | WebAssembly model in a worker |

The model (MobileCLIP-S0 encoders, the trained mask decoder, the Rust/WASM
runtime) and the build script live in the video-system1 repository:
`python3 scripts/build-seg-web.py --site ../vega1-seg [--public]` fills in
`pkg/`, `models/`, `samples/` and `stats.json` (not committed; the built site
is published from the `gh-pages` branch).
