// vega1-seg browser API: "where is the dog?" → a mask of the dog, per frame.
//
//   import { Vega1Seg } from './segment.js';
//   const seg = await Vega1Seg.load();
//   await seg.analyze(video, { frames: 16 });            // encode keyframes once
//   const masks = await seg.masks('the dog');             // per keyframe: { present, bbox, width, height, mask }
//
// Everything runs locally: the video never leaves the page.

const MAX_ANALYSIS_WIDTH = 640;
const MAX_WORKERS = 4;
export const MASK_WIDTH = 192;

class WorkerClient {
  #worker;
  #pending = new Map();
  #nextId = 0;

  constructor() {
    this.#worker = new Worker(new URL('./seg-worker.js', import.meta.url), { type: 'module' });
    this.#worker.onmessage = ({ data }) => {
      const p = this.#pending.get(data.id);
      if (!p) { console.error('stray worker message', JSON.stringify(data).slice(0, 300)); return; }
      this.#pending.delete(data.id);
      if (data.ok) p.resolve(data.result);
      else p.reject(new Error(data.error));
    };
  }

  call(type, payload = {}, transfer = []) {
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#worker.postMessage({ id, type, ...payload }, transfer);
    });
  }
}

/** Unpacks a worker mask: [present, x, y, w, h, width, height, mask...]. */
function unpack(out) {
  const bbox = out[1] < 0 ? null : [out[1], out[2], out[3], out[4]];
  return { present: out[0], bbox, width: out[5], height: out[6], mask: out.subarray(7) };
}

export function isPhone() {
  return navigator.userAgentData?.mobile ?? (/Android|iPhone|iPad|iPod|Mobi/i.test(navigator.userAgent) ||
    (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1));
}

export class Vega1Seg {
  #workers;
  #text = null;
  #texts = new Map(); // query → embedding (precomputed or encoded)
  #owner = new Map(); // keyframe id → worker
  #frames = 0;

  constructor(workers, info) {
    this.#workers = workers;
    this.info = info;
  }

  /** Loads the encoder and mask decoder in `workers` workers (up to 4, 2 on phones). */
  static async load({ baseUrl = new URL('.', import.meta.url).href, workers } = {}) {
    const cores = (navigator.hardwareConcurrency || 2) - 1;
    const n = Math.max(1, Math.min(isPhone() ? 2 : MAX_WORKERS, workers ?? cores));
    const clients = Array.from({ length: n }, () => new WorkerClient());
    const infos = await Promise.all(clients.map((c) => c.call('init', { baseUrl })));
    return new Vega1Seg(clients, { ...infos[0], workers: n });
  }

  get frames() { return this.#frames; }
  get textLoaded() { return this.#text !== null && this.#textReady; }
  #textReady = false;

  /** Loads the text encoder (~90 MB, once) for typed queries. */
  loadText() {
    this.#text ??= this.#workers[0].call('loadText').then(() => { this.#textReady = true; }).catch((err) => {
      this.#text = null;
      throw err;
    });
    return this.#text;
  }

  /** Query embeddings computed ahead (e.g. shipped with the page). */
  addTexts(entries) {
    for (const [query, emb] of entries) this.#texts.set(query.trim().toLowerCase(), emb);
  }

  hasText(query) { return this.#texts.has(query.trim().toLowerCase()); }

  async embed(query) {
    const key = query.trim().toLowerCase();
    if (!this.#texts.has(key)) {
      await this.loadText();
      this.#texts.set(key, new Float32Array(await this.#workers[0].call('embedText', { query: query.trim() })));
    }
    return this.#texts.get(key);
  }

  /**
   * Encodes `frames` keyframes of the video (at the midpoints of equal spans),
   * spread over the workers. Calls `onFrame(i, canvas)` as each is decoded and
   * `onProgress(done, total)` as each is encoded.
   */
  async analyze(video, { frames = 16, onFrame, onProgress } = {}) {
    await Promise.all(this.#workers.map((w) => w.call('drop')));
    this.#owner.clear();
    this.#frames = 0;
    const times = Array.from({ length: frames }, (_, i) => ((i + 0.5) * video.duration) / frames);
    let done = 0;
    await this.#eachFrame(video, times, async (worker, i, rgba, width, height) => {
      await worker.call('encode', { frame: i, rgba: rgba.buffer, width, height }, [rgba.buffer]);
      this.#owner.set(i, worker);
      onProgress?.(++done, frames);
    }, onFrame);
    this.#frames = frames;
    return times;
  }

  /** Masks of a query on every analyzed keyframe (decoder only). */
  async masks(query, { width = MASK_WIDTH } = {}) {
    const text = await this.embed(query);
    return Promise.all(Array.from({ length: this.#frames }, (_, i) =>
      this.#owner.get(i).call('mask', { frame: i, text: Array.from(text), width }).then((r) => unpack(r.out))));
  }

  /**
   * Masks of each query on every frame at `fps` (for export): encodes each
   * frame once, decodes it for every query, keeps no encodings.
   * Returns { times, masks: [query][frame] }. `onProgress(done, total)`.
   */
  async track(video, queries, { fps = 10, width = MASK_WIDTH, onProgress } = {}) {
    const texts = await Promise.all(queries.map(async (q) => Array.from(await this.embed(q))));
    const n = Math.max(1, Math.round(video.duration * fps));
    const times = Array.from({ length: n }, (_, i) => Math.min(video.duration - 0.001, i / fps));
    const out = queries.map(() => new Array(n));
    let done = 0;
    const base = 100000; // ids that don't collide with keyframes
    await this.#eachFrame(video, times, async (worker, i, rgba, w, h) => {
      await worker.call('encode', { frame: base + i, rgba: rgba.buffer, width: w, height: h }, [rgba.buffer]);
      for (const [q, text] of texts.entries()) {
        const forget = q === texts.length - 1;
        out[q][i] = unpack((await worker.call('mask', { frame: base + i, text, width, forget })).out);
      }
      onProgress?.(++done, n);
    });
    return { times, masks: out };
  }

  async #eachFrame(video, times, job, onFrame) {
    const width = Math.min(MAX_ANALYSIS_WIDTH, video.videoWidth) & ~1;
    const height = Math.max(2, Math.round((video.videoHeight * width) / video.videoWidth / 2) * 2);
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const idle = [...this.#workers];
    const waiting = [];
    const acquire = () => (idle.length ? Promise.resolve(idle.pop()) : new Promise((r) => waiting.push(r)));
    const release = (w) => (waiting.length ? waiting.shift()(w) : idle.push(w));
    const jobs = [];
    for (const [i, t] of times.entries()) {
      const worker = await acquire();
      await seek(video, t);
      ctx.drawImage(video, 0, 0, width, height);
      onFrame?.(i, canvas);
      const { data } = ctx.getImageData(0, 0, width, height);
      jobs.push(job(worker, i, data, width, height).finally(() => release(worker)));
    }
    await Promise.all(jobs);
  }
}

/** A hidden <video> for a File/Blob/URL, to read frames from. */
export async function openVideo(src) {
  const url = typeof src === 'string' ? src : URL.createObjectURL(src);
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = url;
  await once(video, 'loadedmetadata');
  return video;
}

function once(el, event) {
  return new Promise((resolve, reject) => {
    el.addEventListener(event, resolve, { once: true });
    el.addEventListener('error', () => reject(new Error(`could not load video (${event})`)), { once: true });
  });
}

async function seek(video, t) {
  video.currentTime = Math.min(t, Math.max(0, video.duration - 0.001));
  await once(video, 'seeked');
}
