// 譯機：瀏覽器內的離線翻譯引擎（手機也能用）
// 在 Web Worker 裡用 Transformers.js 跑 Marian 翻譯模型，避免卡住畫面。
// 模型檔由 Transformers.js 存進瀏覽器的 Cache Storage，下載一次後即可離線使用。
import * as OpenCC from './vendor/opencc/full.js';

// 語言方向 → Hugging Face 上的模型。中日互譯沒有直接模型，經英文轉譯。
const MODELS = {
    'zh>en': 'Xenova/opus-mt-zh-en',
    'en>zh': 'Xenova/opus-mt-en-zh',
    'ja>en': 'Xenova/opus-mt-ja-en',
    'en>ja': 'Kadonox/fugumt-en-ja-onnx',
};
const MAX_LOADED = 2;   // 同時留在記憶體的翻譯模型數，避免手機記憶體不足

// 語音辨識：Whisper base 多語言版。編碼器維持原精度（量化後辨識率掉很多），解碼器用 8 位元量化。
const ASR_MODEL = 'Xenova/whisper-base';
const ASR_DTYPE = { encoder_model: 'fp32', decoder_model_merged: 'q8' };
const ASR_LANG = { zh: 'chinese', ja: 'japanese', en: 'english' };

// 中文模型以簡體訓練：輸入先轉簡體，輸出再轉回臺灣繁體與用語
const toSimplified = OpenCC.Converter({ from: 'tw', to: 'cn' });
const toTraditional = OpenCC.Converter({ from: 'cn', to: 'twp' });

// 離線時讓對外連線立刻失敗，Transformers.js 就只會用快取裡的模型檔，
// 不必等每個檔案連線逾時。必須在載入 Transformers.js 之前裝好。
let offlineMode = false;
const realFetch = self.fetch.bind(self);
self.fetch = (input, init) => {
    const url = new URL(typeof input === 'string' ? input : input.url ?? String(input), self.location.href);
    if (offlineMode && url.origin !== self.location.origin) return Promise.reject(new TypeError('offline'));
    return realFetch(input, init);
};

// 訊息處理要同步掛上，否則載入模組期間送來的訊息會漏掉。
// 一次只處理一件工作，避免同時載入多個模型把記憶體撐爆。
let queue = Promise.resolve();
self.onmessage = ({ data }) => { queue = queue.then(() => handle(data)); };

let pipeline = null;
const ready = import('./vendor/transformers/transformers.min.js').then((tf) => {
    tf.env.allowLocalModels = false;
    // ONNX 的 wasm 改從自己的網域載入，service worker 才能快取
    tf.env.backends.onnx.wasm.wasmPaths = new URL('./vendor/transformers/', import.meta.url).href;
    pipeline = tf.pipeline;
});

const loaded = new Map();   // key → Promise<pipeline>，Map 的順序即最近使用順序

async function getPipe(key, onProgress) {
    if (loaded.has(key)) {
        const p = loaded.get(key);
        loaded.delete(key);
        loaded.set(key, p);
        return p;
    }
    while (loaded.size >= MAX_LOADED) {
        const [oldKey, oldPipe] = loaded.entries().next().value;
        loaded.delete(oldKey);
        oldPipe.then((p) => p.dispose?.()).catch(() => {});
    }
    const files = new Map();
    const p = pipeline('translation', MODELS[key], {
        dtype: 'q8',
        device: 'wasm',
        progress_callback: (e) => {
            if (!onProgress || e.status !== 'progress' || !e.total) return;
            files.set(e.file, [e.loaded, e.total]);
            let done = 0, total = 0;
            for (const [l, t] of files.values()) { done += l; total += t; }
            onProgress(done / total);
        },
    }).catch((err) => { loaded.delete(key); throw err; });
    loaded.set(key, p);
    return p;
}

/** 把各檔案的下載進度合併成 0~1 */
function progressTracker(onProgress) {
    const files = new Map();
    return (e) => {
        if (!onProgress || e.status !== 'progress' || !e.total) return;
        files.set(e.file, [e.loaded, e.total]);
        let done = 0, total = 0;
        for (const [l, t] of files.values()) { done += l; total += t; }
        onProgress(done / total);
    };
}

let asrPipe = null;   // 語音模型獨立保留，不佔翻譯模型的名額
function getAsr(onProgress) {
    asrPipe ??= pipeline('automatic-speech-recognition', ASR_MODEL, {
        dtype: ASR_DTYPE,
        device: 'wasm',
        progress_callback: progressTracker(onProgress),
    }).catch((err) => { asrPipe = null; throw err; });
    return asrPipe;
}

/** audio：16 kHz 單聲道 Float32Array */
async function transcribe(audio, lang) {
    const asr = await getAsr();
    const out = await asr(audio, { language: ASR_LANG[lang], task: 'transcribe', chunk_length_s: 30 });
    const text = (out.text ?? '').trim();
    // Whisper 的中文可能輸出簡體，統一轉成臺灣繁體
    return lang === 'zh' ? toTraditional(text) : text;
}

function route(from, to) {
    if (MODELS[`${from}>${to}`]) return [`${from}>${to}`];
    return [`${from}>en`, `en>${to}`];
}

/** 切成句子再翻，避免超過模型長度上限，也保留換行 */
function splitSentences(line) {
    return line.match(/[^。！？!?]+[。！？!?]*|[。！？!?]+/g)?.map((s) => s.trim()).filter(Boolean) ?? [];
}

async function runStep(key, text) {
    const pipe = await getPipe(key);
    const input = key.startsWith('zh>') ? toSimplified(text) : text;
    const [out] = await pipe(input, { max_new_tokens: 256 });
    const result = out.translation_text.trim();
    return key.endsWith('>zh') ? toTraditional(result) : result;
}

async function translate(text, from, to) {
    const keys = route(from, to);
    const lines = [];
    for (const line of text.split('\n')) {
        const parts = [];
        for (const sentence of splitSentences(line)) {
            let current = sentence;
            for (const key of keys) current = await runStep(key, current);
            parts.push(current);
        }
        lines.push(parts.join(to === 'en' ? ' ' : ''));
    }
    return lines.join('\n').trim();
}

async function handle(data) {
    const { id, type } = data;
    offlineMode = !!data.offline;
    try {
        await ready;
        if (type === 'download') {
            // 逐一載入以觸發下載與快取
            for (let i = 0; i < data.keys.length; i++) {
                const report = (p) => self.postMessage({ id, type: 'progress', value: (i + p) / data.keys.length });
                if (data.keys[i] === 'asr') await getAsr(report);
                else await getPipe(data.keys[i], report);
                report(1);
            }
            self.postMessage({ id, type: 'done' });
        } else if (type === 'transcribe') {
            self.postMessage({ id, type: 'done', result: await transcribe(data.audio, data.lang) });
        } else if (type === 'translate') {
            self.postMessage({ id, type: 'done', result: await translate(data.text, data.from, data.to) });
        }
    } catch (err) {
        self.postMessage({ id, type: 'error', message: String(err?.message ?? err) });
    }
}
