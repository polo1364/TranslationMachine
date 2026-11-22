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
// 實測品質主要掉在解碼器的 8 位元量化（例如「お会計をお願いします」會被翻成 Please check it out.）；
// 解碼器改用 fp16 後譯文與未壓縮版一致，編碼器維持 8 位元量化則沒有影響。
const MT_DTYPE = { encoder_model: 'q8', decoder_model_merged: 'fp16' };
// 例外：英→日模型的 fp16 解碼器檔無法載入（ONNX 判定為無效模型），改用未壓縮的 fp32，只多約 6 MB
const MT_DTYPE_OVERRIDE = { 'en>ja': { encoder_model: 'q8', decoder_model_merged: 'fp32' } };
// 同時留在記憶體的翻譯模型數。中日對話來回各需兩個模型（經英文轉譯），共四個；
// 記憶體充裕的裝置全部留著，免得每換一個人說話就重新載入。手機只留兩個，避免記憶體不足。
const MAX_LOADED = (self.navigator?.deviceMemory ?? 4) >= 8 ? 4 : 2;

// 語音辨識：Whisper 多語言版，兩種大小。
// 實測同一段日語：base 有四處錯字，small 幾乎全對；中文 base 會把「我想買」聽成「我向買」，small 正確。
// 代價是 small 的辨識時間約為 base 的兩倍。兩者用 8 位元量化的辨識結果都與較高精度相同。
const ASR_MODELS = {
    'asr': { id: 'Xenova/whisper-base', dtype: { encoder_model: 'fp32', decoder_model_merged: 'q8' } },
    'asr-small': { id: 'Xenova/whisper-small', dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' } },
    // 即時預覽專用：Whisper 每次都要處理固定 30 秒的視窗，small 一次約 8 秒、base 約 4 秒（桌面實測），
    // 來不及在說話途中出結果；tiny 約 1.5 秒，準確度較粗但只用來預覽，最終結果仍由上面的模型產生。
    'asr-live': { id: 'Xenova/whisper-tiny', dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' } },
};
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
    const p = pipeline('translation', MODELS[key], {
        dtype: MT_DTYPE_OVERRIDE[key] ?? MT_DTYPE,
        device: 'wasm',
        progress_callback: progressTracker(onProgress),
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

// 語音模型獨立保留，不佔翻譯模型的名額。
// 記憶體裡最多同時有：預覽用的 tiny，加上一個主要模型（base 或 small）。
const asrPipes = new Map();
function getAsr(key, onProgress) {
    if (!asrPipes.has(key)) {
        if (key !== 'asr-live') {
            for (const [other, pipe] of asrPipes) {
                if (other === 'asr-live') continue;
                asrPipes.delete(other);
                pipe.then((p) => p.dispose?.()).catch(() => {});
            }
        }
        asrPipes.set(key, pipeline('automatic-speech-recognition', ASR_MODELS[key].id, {
            dtype: ASR_MODELS[key].dtype,
            device: 'wasm',
            progress_callback: progressTracker(onProgress),
        }).catch((err) => { asrPipes.delete(key); throw err; }));
    }
    return asrPipes.get(key);
}

/** audio：16 kHz 單聲道 Float32Array */
async function transcribe(audio, lang, key) {
    const asr = await getAsr(key);
    const out = await asr(audio, { language: ASR_LANG[lang], task: 'transcribe', chunk_length_s: 30 });
    const text = (out.text ?? '').trim();
    // Whisper 的中文可能輸出簡體、標點是半形，統一轉成臺灣繁體與全形標點
    return fixPunctuation(lang === 'zh' ? toTraditional(text) : text, lang);
}

function route(from, to) {
    if (MODELS[`${from}>${to}`]) return [`${from}>${to}`];
    return [`${from}>en`, `en>${to}`];
}

/**
 * 切成句子再翻。這些模型一次只可靠地處理一句，整段丟進去會漏掉後面的句子。
 * 中日文在 。！？ 後切；英文在句點後接空白與大寫／數字／引號時才切，避免切到小數點與縮寫。
 */
function splitSentences(line) {
    return line
        .split(/(?<=[。！？!?])\s*|(?<=\.)\s+(?=[A-Z0-9"'“(])/)
        .map((s) => s.trim())
        .filter(Boolean);
}

/** 模型輸出的中日文常帶半形標點，換成全形 */
function fixPunctuation(text, lang) {
    if (lang === 'zh') {
        return text.replace(/([^\x00-\x7f])\s*([,?!:;])\s*/g, (_, ch, p) => ch + { ',': '，', '?': '？', '!': '！', ':': '：', ';': '；' }[p])
            .replace(/([^\x00-\x7f])\.$/, '$1。');
    }
    if (lang === 'ja') {
        return text.replace(/([^\x00-\x7f])\s*([?!])\s*/g, (_, ch, p) => ch + { '?': '？', '!': '！' }[p]);
    }
    return text;
}

/**
 * 用一個模型翻一段文字。每一步都重新切句：經英文轉譯時，上一步可能把一句翻成兩句，
 * 不重新切的話第二句會被模型丟掉。
 */
async function runStep(key, text) {
    const pipe = await getPipe(key);
    const target = key.slice(-2);
    const parts = [];
    for (const sentence of splitSentences(text)) {
        const input = key.startsWith('zh>') ? toSimplified(sentence) : sentence;
        const [out] = await pipe(input, { max_new_tokens: 256 });
        let result = out.translation_text.trim();
        if (!result) continue;
        result = fixPunctuation(target === 'zh' ? toTraditional(result) : result, target);
        // 中日文沒有空白可分隔句子，缺句尾標點時補上，免得兩句黏在一起
        if (target !== 'en' && !/[。！？…」』）]$/.test(result)) result += '。';
        parts.push(result);
    }
    return parts.join(target === 'en' ? ' ' : '');
}

/** 換成新精度後，舊的量化解碼器檔已用不到，從快取刪掉以釋放空間 */
async function dropStaleFiles() {
    try {
        const cache = await caches.open('transformers-cache');
        for (const request of await cache.keys()) {
            const isOurs = Object.values(MODELS).some((m) => request.url.includes(`/${m}/`));
            if (isOurs && request.url.endsWith('decoder_model_merged_quantized.onnx')) await cache.delete(request);
        }
    } catch { /* 清不掉只是多佔空間，不影響功能 */ }
}

async function translate(text, from, to) {
    const keys = route(from, to);
    const lines = [];
    for (const line of text.split('\n')) {
        let current = line.trim();
        for (const key of keys) {
            if (!current) break;
            current = await runStep(key, current);
        }
        lines.push(current);
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
                if (ASR_MODELS[data.keys[i]]) await getAsr(data.keys[i], report);
                else await getPipe(data.keys[i], report);
                report(1);
            }
            await dropStaleFiles();
            self.postMessage({ id, type: 'done' });
        } else if (type === 'warm') {
            // 預先把模型載入記憶體，第一句話就不用等。個別失敗不影響其他模型。
            for (const key of data.asr ?? []) await getAsr(key).catch(() => {});
            for (const key of (data.mt ?? []).slice(0, MAX_LOADED)) await getPipe(key).catch(() => {});
            self.postMessage({ id, type: 'done' });
        } else if (type === 'transcribe') {
            self.postMessage({ id, type: 'done', result: await transcribe(data.audio, data.lang, data.model) });
        } else if (type === 'translate') {
            self.postMessage({ id, type: 'done', result: await translate(data.text, data.from, data.to) });
        }
    } catch (err) {
        self.postMessage({ id, type: 'error', message: String(err?.message ?? err) });
    }
}
