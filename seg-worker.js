// Runs vega1-seg off the main thread. Protocol:
// main → { id, type, ...payload }, worker → { id, ok, result | error }.
// Every worker holds a Segmenter (encoder + mask decoder) and the encoded
// tiles of the frames it was given; the first worker also loads the text
// encoder for typed queries.
import init, { Segmenter, TextEmbedder } from './pkg/vs1_wasm.js';

const CACHE = 'vega1-seg-models-v1';
const FILES = { vision: 'models/vision_fp16w.onnx', decoder: 'models/seg.onnx', text: 'models/text_fp16w.onnx',
  tokenizer: 'models/tokenizer.json' };

let seg = null;
let text = null;

// Model files are large; keep them in the Cache API across visits when allowed.
async function fetchBytes(url) {
  let cache = null;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    cache = null; // private mode, blocked storage, etc.
  }
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetching ${url}: HTTP ${resp.status}`);
  if (cache) {
    try { await cache.put(url, resp.clone()); } catch { /* quota: still usable, not cached */ }
  }
  return new Uint8Array(await resp.arrayBuffer());
}

const handlers = {
  async init({ baseUrl }) {
    const t0 = performance.now();
    await init();
    self.baseUrl = baseUrl;
    const [vision, decoder] = await Promise.all([FILES.vision, FILES.decoder].map((f) => fetchBytes(new URL(f, baseUrl).href)));
    seg = new Segmenter(vision, decoder);
    return { model: seg.modelName, ms: performance.now() - t0 };
  },
  async loadText() {
    const [onnx, tokenizer] = await Promise.all([FILES.text, FILES.tokenizer].map((f) => fetchBytes(new URL(f, self.baseUrl).href)));
    text = new TextEmbedder(onnx, tokenizer);
    return true;
  },
  embedText({ query }) {
    return text.embedText('', query).slice(0, 512);
  },
  encode({ frame, rgba, width, height }) {
    seg.encodeFrame(frame, new Uint8Array(rgba), width, height);
    return true;
  },
  mask({ frame, text: t, width, forget = false }) {
    const out = seg.mask(frame, new Float32Array(t), width);
    if (forget) seg.dropFrame(frame); // export frames: decoded once, then freed
    return { out };
  },
  drop() {
    seg.dropFrames();
    return true;
  },
};

self.onmessage = async ({ data }) => {
  const { id, type, ...payload } = data;
  try {
    const result = await handlers[type](payload);
    const transfer = result?.out ? [result.out.buffer] : [];
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.message ?? err) });
  }
};
