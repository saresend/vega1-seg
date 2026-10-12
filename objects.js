// Objects within a query's masks, so users can highlight or un-highlight them
// one by one ("the people" → every person is its own object).
//
// Each keyframe's mask (pixels > 0.5) is split into connected regions; regions
// in consecutive keyframes that overlap (or, failing that, sit close) share an
// object id, so a toggle follows the object through the video.

const ON = 0.5;
const MIN_PIXELS = 8; // smaller specks join no object (and always follow their layer)

/** Connected regions (4-neighbour) of mask > ON: labels (0 = none) and regions. */
function regions(mask, w, h) {
  const labels = new Int32Array(w * h);
  const out = [];
  const stack = [];
  for (let start = 0; start < labels.length; start++) {
    if (labels[start] || mask[start] <= ON) continue;
    const label = out.length + 1;
    let n = 0, sx = 0, sy = 0;
    labels[start] = label;
    stack.push(start);
    while (stack.length) {
      const i = stack.pop();
      const x = i % w, y = (i / w) | 0;
      n++; sx += x; sy += y;
      for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]) {
        if (j >= 0 && !labels[j] && mask[j] > ON) {
          labels[j] = label;
          stack.push(j);
        }
      }
    }
    out.push({ label, pixels: n, cx: sx / n / w, cy: sy / n / h });
  }
  return { labels, regions: out };
}

/** Labels grown by up to `steps` pixels over the soft edge (mask > 0.02), so edges follow their region. */
function grow(labels, mask, w, h, steps = 4) {
  let cur = labels;
  for (let s = 0; s < steps; s++) {
    const next = new Int32Array(cur);
    let changed = false;
    for (let i = 0; i < cur.length; i++) {
      if (cur[i] || mask[i] <= 0.02) continue;
      const x = i % w;
      const l = (x > 0 && cur[i - 1]) || (x < w - 1 && cur[i + 1]) || (i >= w && cur[i - w]) || (i + w < cur.length && cur[i + w]) || 0;
      if (l) { next[i] = l; changed = true; }
    }
    cur = next;
    if (!changed) break;
  }
  return cur;
}

/**
 * Splits a result's keyframe masks into objects. Adds to each frame
 * `labels` and `objects` (label → object id, 0 for specks), and returns the
 * number of objects.
 */
export function findObjects(result) {
  const { width: w, height: h, frames } = result;
  let next = 1;
  let prev = null;
  for (const f of frames) {
    const { labels, regions: regs } = regions(f.mask, w, h);
    const objects = new Int32Array(regs.length + 1);
    for (const r of regs) {
      if (r.pixels < MIN_PIXELS) continue;
      let id = 0;
      if (prev) {
        // The previous keyframe's object this region overlaps most.
        const counts = new Map();
        for (let i = 0; i < labels.length; i++) {
          if (labels[i] === r.label && prev.labels[i]) {
            const o = prev.objects[prev.labels[i]];
            if (o) counts.set(o, (counts.get(o) ?? 0) + 1);
          }
        }
        let best = 0;
        for (const [o, c] of counts) if (c > best) { best = c; id = o; }
        if (!id) {
          // No overlap (fast motion): the nearest previous object, if close.
          let d = 0.12;
          for (const q of prev.regs) {
            const o = prev.objects[q.label];
            const dist = Math.hypot(q.cx - r.cx, q.cy - r.cy);
            if (o && dist < d) { d = dist; id = o; }
          }
        }
      }
      objects[r.label] = id || next++;
    }
    Object.assign(f, { labels, owner: grow(labels, f.mask, w, h), objects, regs });
    prev = { labels, objects, regs };
  }
  // Objects to count for people: big enough, or seen in two keyframes (the
  // rest are specks; they can still be toggled).
  const pixels = new Map(), seen = new Map();
  for (const f of frames) {
    for (const r of f.regs) {
      const o = f.objects[r.label];
      if (!o) continue;
      pixels.set(o, (pixels.get(o) ?? 0) + r.pixels);
      seen.set(o, (seen.get(o) ?? 0) + 1);
    }
  }
  result.mainObjects = new Set([...pixels.keys()].filter((o) => seen.get(o) >= 2 || pixels.get(o) >= 40));
  result.objectCount = result.mainObjects.size;
  result.hiddenObjects ??= new Set();
  result.filtered = null;
  return result.objectCount;
}

/** The object id under normalized point (x, y) in keyframe `f`, searching a few pixels around. */
export function objectAt(result, f, x, y) {
  const { width: w, height: h } = result;
  const frame = result.frames[f];
  const px = Math.round(x * w - 0.5), py = Math.round(y * h - 0.5);
  for (let r = 0; r <= 3; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const xx = px + dx, yy = py + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const o = frame.objects[frame.owner[yy * w + xx]];
        if (o) return o;
      }
    }
  }
  return 0;
}

/**
 * Keyframe masks split by the user's choice: `shown` (highlighted objects and
 * specks) and `hidden` (un-highlighted objects), with the shown mask's box.
 * Cached until the choice changes.
 */
export function filtered(result) {
  if (result.filtered) return result.filtered;
  const { width: w, height: h } = result;
  const hide = result.hiddenObjects;
  result.filtered = result.frames.map((f) => {
    if (!hide.size) return { shown: f.mask, hidden: null, bbox: f.bbox, present: f.present };
    const shown = new Float32Array(f.mask.length);
    const hidden = new Float32Array(f.mask.length);
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (let i = 0; i < shown.length; i++) {
      const off = hide.has(f.objects[f.owner[i]]);
      if (off) hidden[i] = f.mask[i];
      else {
        shown[i] = f.mask[i];
        if (f.mask[i] > ON) {
          const x = i % w, y = (i / w) | 0;
          if (x < x0) x0 = x;
          if (y < y0) y0 = y;
          if (x > x1) x1 = x;
          if (y > y1) y1 = y;
        }
      }
    }
    const bbox = x1 < 0 ? null : [x0 / w, y0 / h, (x1 + 1 - x0) / w, (y1 + 1 - y0) / h];
    return { shown, hidden, bbox, present: bbox ? f.present : 0 };
  });
  return result.filtered;
}

export function toggleObject(result, id) {
  if (result.hiddenObjects.has(id)) result.hiddenObjects.delete(id);
  else result.hiddenObjects.add(id);
  result.filtered = null;
}

/**
 * Applies a result's object choice to a mask of another frame (an export
 * frame): each of its regions is kept unless most of it falls on hidden
 * objects of the nearest keyframe.
 */
export function applyChoice(result, mask, w, h, t, duration) {
  if (!result.hiddenObjects.size) return mask;
  const n = result.frames.length;
  const f = Math.min(n - 1, Math.max(0, Math.round((t / duration) * n - 0.5)));
  const key = result.frames[f];
  const kw = result.width, kh = result.height;
  const hide = result.hiddenObjects;
  const { labels, regions: regs } = regions(mask, w, h);
  const votes = new Map(); // label → [hidden, shown]
  for (let i = 0; i < labels.length; i++) {
    if (!labels[i]) continue;
    const x = Math.min(kw - 1, Math.floor(((i % w) + 0.5) / w * kw));
    const y = Math.min(kh - 1, Math.floor((((i / w) | 0) + 0.5) / h * kh));
    const o = key.objects[key.labels[y * kw + x]];
    if (!o) continue;
    const v = votes.get(labels[i]) ?? [0, 0];
    v[hide.has(o) ? 0 : 1]++;
    votes.set(labels[i], v);
  }
  const drop = new Set(regs.filter((r) => { const v = votes.get(r.label); return v && v[0] > v[1]; }).map((r) => r.label));
  if (!drop.size) return mask;
  const owner = grow(labels, mask, w, h);
  const out = new Float32Array(mask);
  for (let i = 0; i < out.length; i++) if (drop.has(owner[i])) out[i] = 0;
  return out;
}
