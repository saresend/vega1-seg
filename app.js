// Vega1-seg page. Samples ship with masks for their example queries (instant);
// typed queries and your own videos load the model (~25 MB) and the text
// encoder (~90 MB) on first use, analyze 16 keyframes once, then answer each
// new query with the small decoder only.
//
// Every answered query is a mask layer that can be highlighted or hidden, and
// each mask splits into objects (objects.js) that a click on the video
// highlights or un-highlights. Views and exports show only what is highlighted.
import { Vega1Seg, isPhone, openVideo } from './segment.js';
import { applyChoice, filtered, findObjects, objectAt, toggleObject } from './objects.js';

const $ = (id) => document.getElementById(id);
const escape = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const ms = (t) => (t < 1000 ? `${Math.round(t)} ms` : `${(t / 1000).toFixed(1)} s`);
const GOLD = [204, 164, 59];
const KEYFRAMES = isPhone() ? 8 : 16;
const EXPORT_MASK_WIDTH = 320;

let seg = null; // the model, once loaded
let segLoading = null;
let clip = null; // { blob, name, sample, hidden: <video> for reading frames, analyzed: bool }
// query → { frames: [{ present, bbox, mask, labels, owner, objects }], width, height, visible, hiddenObjects }
const results = new Map();
let active = null; // the latest query (timeline, answer)
let mode = 'overlay';
const video = $('video');
const overlay = $('overlay');
const octx = overlay.getContext('2d');
let presets = []; // example queries with shipped text embeddings
const askTimes = [];

// ---------------------------------------------------------------- status

function status(state, text) {
  $('status').dataset.state = state;
  $('status-text').textContent = text;
}
function working(text, frac = null) {
  $('working').hidden = text === null;
  if (text !== null) $('working-text').textContent = text;
  $('working-bar').style.width = frac === null ? '0' : `${Math.round(frac * 100)}%`;
}
function answer(html) { $('answer').innerHTML = html; }

// ---------------------------------------------------------------- model

function loadModel() {
  segLoading ??= (async () => {
    status('busy', 'the model loads (25 MB)…');
    const t0 = performance.now();
    seg = await Vega1Seg.load();
    seg.addTexts(presetTexts);
    status('ready', `ready (${ms(performance.now() - t0)})`);
    return seg;
  })().catch((err) => {
    segLoading = null;
    status('error', 'the model did not load');
    throw err;
  });
  return segLoading;
}

function wantText() {
  loadModel().then((s) => {
    if (s.textLoaded) return;
    $('text-status').textContent = 'the text model loads (90 MB, one time only)…';
    return s.loadText().then(() => { $('text-status').textContent = ''; });
  }).catch(() => { $('text-status').textContent = 'the text model did not load'; });
}

// Shipped text embeddings for the example queries (samples/texts.*).
let presetTexts = [];
async function loadPresetTexts() {
  const [keys, data] = await Promise.all([
    fetch('samples/texts.json').then((r) => (r.ok ? r.json() : null)),
    fetch('samples/texts.f32').then((r) => (r.ok ? r.arrayBuffer() : null)),
  ]).catch(() => [null, null]);
  if (!keys || !data) return;
  const all = new Float32Array(data);
  // embed-texts keys are [context, focus] pairs; the query is the focus.
  presetTexts = keys.map(([, q], i) => [q, all.slice(i * 513, i * 513 + 512)]).filter(([q]) => q !== 'a video');
}

// ---------------------------------------------------------------- clips

const precomputed = new Map(); // sample file → [[query, result]] (kept so their choices survive a switch)

async function setClip(blob, name, sample = null) {
  results.clear();
  active = null;
  clip = { blob, name, sample, hidden: null, analyzed: false };
  $('drop').hidden = true;
  video.src = URL.createObjectURL(blob);
  video.play().catch(() => {});
  $('play').textContent = '❚❚';
  $('clip-name').textContent = sample?.title ?? name;
  $('clip-credit').innerHTML = sample?.credit
    ? sample.source ? `<a href="${escape(sample.source)}" target="_blank" rel="noopener">${escape(sample.credit)}</a>` : escape(sample.credit)
    : 'your video · it stays on this device';
  for (const c of document.querySelectorAll('.sample')) c.classList.toggle('active', c === sample?.card);
  renderAll();
  const first = sample?.queries?.[0];
  if (first) find(first);
  else answer('<span class="muted">Write what to find, for example "the person".</span>');
}

function renderChips() {
  const queries = clip?.sample?.queries ?? presets;
  $('chips').replaceChildren(...queries.slice(0, 8).map((q) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = q;
    b.setAttribute('aria-pressed', String(results.get(q)?.visible ?? false));
    b.addEventListener('click', () => { $('query').value = q; find(q); });
    return b;
  }));
}

// ---------------------------------------------------------------- finding

/** A query's masks: shipped with the sample, already found, or from the model. */
async function masksFor(query) {
  if (results.has(query)) return { r: results.get(query), analyzedNow: false };
  const shipped = clip.sample && precomputed.get(clip.sample.file)?.find(([q]) => q === query)?.[1];
  if (shipped) return { r: shipped, analyzedNow: false };
  const s = await loadModel();
  if (!s.hasText(query) && !s.textLoaded) $('text-status').textContent = 'the text model loads (90 MB, one time only)…';
  let analyzedNow = false;
  if (!clip.analyzed) {
    clip.hidden ??= await openVideo(clip.blob);
    const forClip = clip;
    working(`the model examines ${KEYFRAMES} frames…`, 0);
    await s.analyze(clip.hidden, { frames: KEYFRAMES, onProgress: (d, n) => working(`the model examines the frames: ${d} of ${n}`, d / n) });
    if (forClip !== clip) throw new Error('the video changed');
    clip.analyzed = true;
    analyzedNow = true;
  }
  working('the model makes the masks…', 1);
  const t1 = performance.now();
  const frames = await s.masks(query);
  $('text-status').textContent = '';
  if (!analyzedNow) recordAsk(performance.now() - t1);
  return { r: { frames, width: frames[0].width, height: frames[0].height }, analyzedNow };
}

let findToken = 0;
async function find(query) {
  query = query.trim().replace(/^where (is|are)\s+/i, '').replace(/\?$/, '');
  if (!query || !clip) return;
  const token = ++findToken;
  const forClip = clip;
  $('query').value = query;
  const t0 = performance.now();
  try {
    const { r, analyzedNow } = await masksFor(query);
    if (token !== findToken || forClip !== clip) return;
    if (r.objectCount === undefined) findObjects(r);
    r.visible = true;
    results.delete(query); // newest last
    results.set(query, r);
    active = query;
    working(null);
    showAnswer(query, r, performance.now() - t0, analyzedNow);
  } catch (err) {
    working(null);
    if (err.message !== 'the video changed') answer(`<span class="err">${escape(err.message)}</span>`);
  }
  renderAll();
}

function showAnswer(query, r, took, analyzedNow) {
  const n = r.frames.length;
  const found = r.frames.filter((f) => f.present > 0.5 && f.bbox).length;
  const sure = Math.max(...r.frames.map((f) => f.present));
  const time = analyzedNow ? `examined the video and answered in ${ms(took)}` : took < 20 ? 'instant' : ms(took);
  const objects = r.objectCount > 1 ? ` · ${r.objectCount} objects` : '';
  answer(found
    ? `<span class="big">${escape(query)}</span><br>found in <b>${found}</b> of ${n} frames${objects} · ${Math.round(sure * 100)}% sure · ${time}`
    : `<span class="big">not found</span><br>the model does not see ${escape(query)} in this video · ${time}`);
}

function recordAsk(t) {
  askTimes.push(t);
  const mean = askTimes.reduce((a, b) => a + b, 0) / askTimes.length;
  $('stat-speed').textContent = ms(mean);
  $('stat-speed-label').textContent = `average time for each new question on this device (${askTimes.length})`;
}

// ---------------------------------------------------------------- layers

function renderLayers() {
  const rows = [...results.entries()].reverse();
  $('layers-card').hidden = !rows.length;
  $('layers').replaceChildren(...rows.map(([q, r]) => {
    const li = document.createElement('li');
    li.className = `layer${r.visible ? '' : ' off'}${q === active ? ' current' : ''}`;
    const hidden = r.hiddenObjects.size;
    const found = r.frames.some((f) => f.present > 0.5 && f.bbox);
    const shownCount = r.objectCount - [...r.hiddenObjects].filter((id) => r.mainObjects.has(id)).length;
    const info = !found ? 'not found'
      : r.objectCount > 1 ? `${shownCount} of ${r.objectCount} objects highlighted` : hidden ? 'un-highlighted' : '1 object';
    li.innerHTML = `<button class="eye" aria-pressed="${r.visible}" aria-label="${r.visible ? 'Hide' : 'Highlight'} ${escape(q)}" title="${r.visible ? 'hide' : 'highlight'}"><i></i></button>` +
      `<div class="what"><span class="name">${escape(q)}</span><span class="info">${info}${hidden ? ' · <button class="all">highlight all</button>' : ''}</span></div>` +
      `<button class="x" aria-label="Remove ${escape(q)}" title="remove">×</button>`;
    li.querySelector('.eye').addEventListener('click', () => { r.visible = !r.visible; renderAll(); });
    li.querySelector('.all')?.addEventListener('click', () => { r.hiddenObjects.clear(); r.filtered = null; renderAll(); });
    li.querySelector('.x').addEventListener('click', () => {
      results.delete(q);
      if (active === q) active = [...results.keys()].pop() ?? null;
      renderAll();
    });
    li.querySelector('.name').addEventListener('click', () => { active = q; renderAll(); });
    return li;
  }));
}

function renderAll() {
  renderLayers();
  renderChips();
  renderTimeline();
  setExports();
}

// ---------------------------------------------------------------- drawing

const visibleLayers = () => [...results.entries()].filter(([, r]) => r.visible);

/** Keyframe position at the video's current time: [i0, i1, weight of i1]. */
function keyPosition(n) {
  const k = Math.min(n - 1, Math.max(0, (video.currentTime / video.duration) * n - 0.5));
  const i0 = Math.floor(k);
  return [i0, Math.min(n - 1, i0 + 1), k - i0];
}

/** One layer at the current time: highlighted and un-highlighted masks, box, presence. */
function layerNow(query, r) {
  const f = filtered(r);
  const [i0, i1, a] = keyPosition(r.frames.length);
  const blend = (x, y) => {
    if (!x && !y) return null;
    const out = new Float32Array(r.width * r.height);
    for (let i = 0; i < out.length; i++) out[i] = (x ? x[i] : 0) * (1 - a) + (y ? y[i] : 0) * a;
    return out;
  };
  const b0 = f[i0].bbox, b1 = f[i1].bbox;
  const bbox = b0 && b1 ? b0.map((v, j) => v * (1 - a) + b1[j] * a) : (a < 0.5 ? b0 : b1);
  return { query, width: r.width, height: r.height, shown: blend(f[i0].shown, f[i1].shown), hidden: blend(f[i0].hidden, f[i1].hidden),
    bbox, present: f[i0].present * (1 - a) + f[i1].present * a };
}

const smooth = (p) => Math.min(1, Math.max(0, (p - 0.3) / 0.4));

/** A mask as an image: `kind` overlay (gold, outlined), ghost (faint outline), matte (white) or alpha. */
function maskImage(mask, w, h, kind) {
  const img = new ImageData(w, h);
  const d = img.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, p = mask[i], o = i * 4;
      if (kind === 'overlay' || kind === 'ghost') {
        const edge = p >= 0.5 && ((x > 0 && mask[i - 1] < 0.5) || (x < w - 1 && mask[i + 1] < 0.5) ||
          (y > 0 && mask[i - w] < 0.5) || (y < h - 1 && mask[i + w] < 0.5));
        const c = kind === 'overlay' ? GOLD : [255, 255, 255];
        d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2];
        d[o + 3] = kind === 'overlay' ? (edge ? 255 : smooth(p) * 120) : (edge ? 150 : smooth(p) * 28);
      } else {
        d[o] = d[o + 1] = d[o + 2] = 255;
        d[o + 3] = smooth(p) * 255;
      }
    }
  }
  const c = new OffscreenCanvas(w, h);
  c.getContext('2d').putImageData(img, 0, 0);
  return c;
}

/** The union (max) of the layers' highlighted masks, at the first layer's size. */
function union(layers) {
  const shown = layers.filter((l) => l.shown);
  if (!shown.length) return null;
  const { width: w, height: h } = shown[0];
  const out = new Float32Array(w * h);
  for (const l of shown) {
    if (l.width === w && l.height === h) for (let i = 0; i < out.length; i++) out[i] = Math.max(out[i], l.shown[i]);
  }
  return { mask: out, width: w, height: h };
}

/** Draws a view of the layers over a frame into `ctx` at rect (x, y, w, h). */
function drawView(ctx, layers, kind, rect, source) {
  const [x, y, w, h] = rect;
  ctx.imageSmoothingEnabled = true;
  const u = union(layers);
  if (kind === 'matte') {
    ctx.fillStyle = '#000';
    ctx.fillRect(x, y, w, h);
    if (u) ctx.drawImage(maskImage(u.mask, u.width, u.height, 'matte'), x, y, w, h);
    return;
  }
  if (kind === 'cutout') {
    if (!u) return;
    const tmp = new OffscreenCanvas(Math.round(w), Math.round(h));
    const t = tmp.getContext('2d');
    t.drawImage(source, 0, 0, tmp.width, tmp.height);
    t.globalCompositeOperation = 'destination-in';
    t.drawImage(maskImage(u.mask, u.width, u.height, 'alpha'), 0, 0, tmp.width, tmp.height);
    ctx.drawImage(tmp, x, y);
    return;
  }
  for (const l of layers) {
    if (l.hidden) ctx.drawImage(maskImage(l.hidden, l.width, l.height, 'ghost'), x, y, w, h);
    if (kind === 'overlay' && l.shown) ctx.drawImage(maskImage(l.shown, l.width, l.height, 'overlay'), x, y, w, h);
    if (l.bbox && l.present > 0.5) {
      const [bx, by, bw, bh] = l.bbox;
      ctx.strokeStyle = 'rgb(204,164,59)';
      ctx.lineWidth = Math.max(2, w / 400);
      ctx.setLineDash(kind === 'box' ? [] : [8, 6]);
      ctx.strokeRect(x + bx * w, y + by * h, bw * w, bh * h);
      if (kind === 'box') {
        const label = `${l.query} ${Math.round(l.present * 100)}%`;
        ctx.font = `700 ${Math.max(12, w / 60)}px ui-monospace, Menlo, monospace`;
        const tw = ctx.measureText(label).width + 10;
        const ty = Math.max(y, y + by * h - 22);
        ctx.fillStyle = 'rgb(204,164,59)';
        ctx.fillRect(x + bx * w, ty, tw, 22);
        ctx.fillStyle = '#242F40';
        ctx.fillText(label, x + bx * w + 5, ty + 16);
      }
    }
  }
  ctx.setLineDash([]);
}

function contentRect() {
  const W = overlay.width, H = overlay.height;
  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  const s = Math.min(W / vw, H / vh);
  return [(W - vw * s) / 2, (H - vh * s) / 2, vw * s, vh * s];
}

const currentLayers = () => visibleLayers().map(([q, r]) => layerNow(q, r));

function frame() {
  const dpr = devicePixelRatio || 1;
  const W = Math.round(overlay.clientWidth * dpr), H = Math.round(overlay.clientHeight * dpr);
  if (overlay.width !== W || overlay.height !== H) { overlay.width = W; overlay.height = H; }
  octx.clearRect(0, 0, W, H);
  const layers = video.duration && video.readyState >= 2 ? currentLayers() : [];
  $('stage').classList.toggle('checker', mode === 'cutout' && layers.length > 0);
  if (layers.length) drawView(octx, layers, mode, contentRect(), video);
  if (video.duration) $('playhead').style.left = `${(video.currentTime / video.duration) * 100}%`;
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function renderTimeline() {
  const r = results.get(active);
  const f = r ? filtered(r) : [];
  $('cells').replaceChildren(...f.map((x) => {
    const s = document.createElement('span');
    s.style.height = `${Math.max(4, x.present * 100)}%`;
    if (x.present > 0.5 && x.bbox && r.visible) s.className = 'on';
    return s;
  }));
}
$('timeline').addEventListener('click', (e) => {
  const r = $('timeline').getBoundingClientRect();
  if (video.duration) video.currentTime = ((e.clientX - r.left) / r.width) * video.duration;
});

// ---------------------------------------------------------------- picking objects

/** The visible layer and object under a client point (newest layer first), or null. */
function pick(clientX, clientY) {
  if (!video.duration) return null;
  const box = overlay.getBoundingClientRect();
  const dpr = overlay.width / box.width;
  const [x, y, w, h] = contentRect();
  const nx = ((clientX - box.left) * dpr - x) / w, ny = ((clientY - box.top) * dpr - y) / h;
  if (nx < 0 || ny < 0 || nx > 1 || ny > 1) return null;
  for (const [q, r] of visibleLayers().reverse()) {
    const [i0, i1, a] = keyPosition(r.frames.length);
    const id = objectAt(r, a < 0.5 ? i0 : i1, nx, ny) || objectAt(r, a < 0.5 ? i1 : i0, nx, ny);
    if (id) return { query: q, r, id };
  }
  return null;
}

const stage = $('stage');
stage.addEventListener('click', (e) => {
  if (e.target.closest('.stage-bar, .drop')) return;
  const hit = pick(e.clientX, e.clientY);
  if (!hit) return;
  toggleObject(hit.r, hit.id);
  active = hit.query;
  renderAll();
});
stage.addEventListener('mousemove', (e) => {
  stage.classList.toggle('pick', !e.target.closest('.stage-bar') && !!pick(e.clientX, e.clientY));
});

for (const b of document.querySelectorAll('[data-mode]')) {
  b.addEventListener('click', () => {
    mode = b.dataset.mode;
    for (const o of document.querySelectorAll('[data-mode]')) o.setAttribute('aria-pressed', String(o === b));
  });
}
$('play').addEventListener('click', () => {
  if (video.paused) { video.play(); $('play').textContent = '❚❚'; $('play').ariaLabel = 'Pause'; }
  else { video.pause(); $('play').textContent = '▶'; $('play').ariaLabel = 'Play'; }
});

// ---------------------------------------------------------------- export

function setExports() {
  const ready = !!(clip && visibleLayers().length);
  for (const id of ['exp-matte', 'exp-boxes', 'exp-frame']) $(id).disabled = !ready;
}

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}
const slug = (s) => s.replace(/\.[^.]+$/, '').replace(/\W+/g, '-');
const stem = () => `${slug(clip?.sample?.title ?? clip?.name ?? 'video')}-${visibleLayers().map(([q]) => slug(q)).join('+')}`;

/** Every frame at the chosen rate: the visible layers' masks with their object choices applied. */
async function tracked() {
  const s = await loadModel();
  clip.hidden ??= await openVideo(clip.blob);
  const fps = Number($('fps').value);
  const layers = visibleLayers();
  working('export: the model examines every frame…', 0);
  const t = await s.track(clip.hidden, layers.map(([q]) => q), {
    fps, width: EXPORT_MASK_WIDTH, onProgress: (d, n) => working(`export: frame ${d} of ${n}`, d / n),
  });
  const duration = clip.hidden.duration;
  const perLayer = layers.map(([q, r], li) => t.masks[li].map((m, i) => {
    const mask = applyChoice(r, m.mask, m.width, m.height, t.times[i], duration);
    return { ...m, query: q, shown: mask, hidden: null, bbox: mask === m.mask ? m.bbox : boxOf(mask, m.width, m.height) };
  }));
  return { times: t.times, fps, frames: t.times.map((_, i) => perLayer.map((l) => l[i])) };
}

function boxOf(mask, w, h) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] <= 0.5) continue;
    const x = i % w, y = (i / w) | 0;
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  return x1 < 0 ? null : [x0 / w, y0 / h, (x1 + 1 - x0) / w, (y1 + 1 - y0) / h];
}

const fail = (err) => { working(null); answer(`<span class="err">${escape(err.message)}</span>`); };

$('exp-boxes').addEventListener('click', async () => {
  try {
    const { times, fps, frames } = await tracked();
    working(null);
    const out = { fps, width: clip.hidden.videoWidth, height: clip.hidden.videoHeight,
      queries: visibleLayers().map(([q]) => q),
      frames: times.map((t, i) => ({ t: +t.toFixed(3), objects: frames[i].map((m) => ({ query: m.query, present: +m.present.toFixed(4),
        bbox: m.present > 0.5 && m.bbox ? m.bbox.map((v) => +v.toFixed(4)) : null })) })) };
    download(new Blob([JSON.stringify(out, null, 1)], { type: 'application/json' }), `${stem()}-boxes.json`);
  } catch (err) { fail(err); }
});

$('exp-matte').addEventListener('click', async () => {
  try {
    const { frames, fps } = await tracked();
    const v = clip.hidden;
    const W = Math.min(1280, v.videoWidth) & ~1, H = Math.round((v.videoHeight * W) / v.videoWidth / 2) * 2;
    const canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    const type = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find((t) => MediaRecorder.isTypeSupported(t));
    const rec = new MediaRecorder(canvas.captureStream(fps), { mimeType: type, videoBitsPerSecond: 8e6 });
    const chunks = [];
    rec.ondataavailable = (e) => chunks.push(e.data);
    const stopped = new Promise((r) => { rec.onstop = r; });
    rec.start();
    for (const [i, layers] of frames.entries()) {
      drawView(ctx, layers.filter((m) => m.present > 0.05), 'matte', [0, 0, W, H]);
      working(`export: the video is written, frame ${i + 1} of ${frames.length}`, (i + 1) / frames.length);
      await new Promise((r) => setTimeout(r, 1000 / fps));
    }
    rec.stop();
    await stopped;
    working(null);
    download(new Blob(chunks, { type: type.split(';')[0] }), `${stem()}-matte.${type.includes('mp4') ? 'mp4' : 'webm'}`);
  } catch (err) { fail(err); }
});

$('exp-frame').addEventListener('click', () => {
  const W = video.videoWidth, H = video.videoHeight;
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  if (mode !== 'matte' && mode !== 'cutout') ctx.drawImage(video, 0, 0, W, H);
  // The file shows the highlighted objects only (no faint outlines of the others).
  drawView(ctx, currentLayers().map((l) => ({ ...l, hidden: null })), mode, [0, 0, W, H], video);
  canvas.toBlob((b) => download(b, `${stem()}-${mode}-${video.currentTime.toFixed(2)}s.png`), 'image/png');
});

// ---------------------------------------------------------------- inputs

$('ask').addEventListener('submit', (e) => {
  e.preventDefault();
  find($('query').value);
  $('query').blur();
});
$('query').addEventListener('focus', wantText);
$('yours').addEventListener('click', () => $('file').click());
$('browse').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (f) setClip(f, f.name);
  e.target.value = '';
});
stage.addEventListener('dragover', (e) => { e.preventDefault(); stage.classList.add('over'); });
stage.addEventListener('dragleave', () => stage.classList.remove('over'));
stage.addEventListener('drop', (e) => {
  e.preventDefault();
  stage.classList.remove('over');
  const f = [...e.dataTransfer.files].find((x) => x.type.startsWith('video/'));
  if (f) setClip(f, f.name);
});

// ---------------------------------------------------------------- samples

/** samples/samples.json (scripts/build-seg-web.py): clips with masks for their example queries. */
async function loadSamples() {
  const manifest = await fetch('samples/samples.json').then((r) => (r.ok ? r.json() : [])).catch(() => []);
  $('samples').replaceChildren(...manifest.map((entry) => {
    const b = document.createElement('button');
    b.className = 'sample';
    b.type = 'button';
    b.title = entry.credit ?? '';
    const tile = Math.max(0, Math.round(entry.strip_frames / 2) - 1);
    b.innerHTML = `<span class="thumb" style="background-image:url('samples/${encodeURI(entry.strip)}');` +
      `background-size:${entry.strip_frames * 100}% 100%;background-position:${(tile / (entry.strip_frames - 1)) * 100}% 0"></span>` +
      `<span>${escape(entry.title)}</span>`;
    b.addEventListener('click', () => openSample(entry));
    entry.card = b;
    return b;
  }));
  return manifest;
}

async function openSample(entry) {
  if (!precomputed.has(entry.file)) {
    const [blob, info, bytes] = await Promise.all([
      fetch(`samples/${encodeURI(entry.file)}`).then((r) => r.blob()),
      fetch(`samples/${encodeURI(entry.seg)}`).then((r) => r.json()),
      fetch(`samples/${encodeURI(entry.masks)}`).then((r) => r.arrayBuffer()),
    ]);
    const u8 = new Uint8Array(bytes);
    const size = info.mask_width * info.mask_height;
    entry.blob = blob;
    precomputed.set(entry.file, info.queries.map((q, qi) => [q.query, {
      width: info.mask_width, height: info.mask_height,
      frames: q.frames.map((f, fi) => {
        const off = (qi * info.frames + fi) * size;
        const mask = new Float32Array(size);
        for (let i = 0; i < size; i++) mask[i] = u8[off + i] / 255;
        return { present: f.present, bbox: f.bbox, mask };
      }),
    }]));
  }
  setClip(entry.blob, entry.file, entry);
}

// ---------------------------------------------------------------- start

(async () => {
  status('busy', 'loads…');
  const [manifest] = await Promise.all([loadSamples(), loadPresetTexts()]);
  presets = presetTexts.map(([q]) => q);
  $('credits').textContent = `Sample videos: ${[...new Set(manifest.map((m) => m.credit_short).filter(Boolean))].join(' · ') || 'none'}`;
  fetch('stats.json').then((r) => (r.ok ? r.json() : null)).then((s) => {
    if (!s) return;
    $('stat-iou').textContent = `${Math.round(s.miou * 100)}%`;
    $('stat-absent').textContent = `${Math.round(s.absent * 100)}%`;
  }).catch(() => {});
  status('ready', 'ready');
  if (manifest.length) await openSample(manifest[0]);
  else $('drop').hidden = false;
})();

// For automated tests: the layers and the point of an object in the current frame.
window.vega1seg = {
  layers: () => [...results.entries()].map(([q, r]) => ({ query: q, visible: r.visible, objects: r.objectCount, hidden: [...r.hiddenObjects] })),
  objectPoint(query, n = 0) {
    const r = results.get(query);
    const [i0, i1, a] = keyPosition(r.frames.length);
    const f = r.frames[a < 0.5 ? i0 : i1];
    const reg = f.regs.filter((g) => f.objects[g.label]).sort((p, q) => q.pixels - p.pixels)[n];
    if (!reg) return null;
    const box = overlay.getBoundingClientRect(), [x, y, w, h] = contentRect(), dpr = overlay.width / box.width;
    return { x: box.left + (x + reg.cx * w) / dpr, y: box.top + (y + reg.cy * h) / dpr, id: f.objects[reg.label] };
  },
};
