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

// ---------- 專有名詞詞彙表 ----------
// 這類小模型常把專有名詞翻錯（捷運站→急行、牛肉麵→牛パスタ）。做法：翻譯前把已知的詞換成代號，
// 翻完再把代號換成正確的譯名。代號用三個大寫字母：實測 8 個候選中這 6 個能原樣通過
// 中→英→日、日→英→中兩段轉譯（ZKW、WQJ 會被改掉，不用）。
// 直接把英文譯名塞進中日文句子則不可行，模型會把它改得面目全非（MRT station → MRT verse）。
const PLACEHOLDERS = ['QZX', 'WVK', 'XJQ', 'VQX', 'JXZ', 'KZQ'];

// 內建詞彙：[中文（臺灣）, 日文, 英文]。使用者自訂的詞彙由頁面傳入，優先於內建。
const BUILTIN_GLOSSARY = [
    // 交通
    ['捷運站', 'MRT駅', 'MRT station'], ['捷運', 'MRT', 'MRT'], ['高鐵', '台湾高速鉄道', 'Taiwan High Speed Rail'],
    ['台鐵', '台湾鉄道', 'Taiwan Railways'], ['悠遊卡', '悠遊カード', 'EasyCard'], ['新幹線', '新幹線', 'Shinkansen'],
    ['地下鐵', '地下鉄', 'subway'], ['地鐵', '地下鉄', 'subway'], ['西瓜卡', 'Suica', 'Suica'],
    // 臺灣地名與景點
    ['台北車站', '台北駅', 'Taipei Main Station'], ['臺北車站', '台北駅', 'Taipei Main Station'], ['台北101', '台北101', 'Taipei 101'],
    ['西門町', '西門町', 'Ximending'], ['九份', '九份', 'Jiufen'], ['淡水', '淡水', 'Tamsui'],
    ['士林夜市', '士林夜市', 'Shilin Night Market'], ['夜市', '夜市', 'night market'],
    ['故宮博物院', '故宮博物院', 'National Palace Museum'], ['桃園機場', '桃園空港', 'Taoyuan Airport'],
    ['松山機場', '松山空港', 'Songshan Airport'], ['日月潭', '日月潭', 'Sun Moon Lake'], ['阿里山', '阿里山', 'Alishan'],
    ['台北', '台北', 'Taipei'], ['臺北', '台北', 'Taipei'], ['台中', '台中', 'Taichung'], ['台南', '台南', 'Tainan'],
    ['高雄', '高雄', 'Kaohsiung'], ['花蓮', '花蓮', 'Hualien'], ['台灣', '台湾', 'Taiwan'], ['臺灣', '台湾', 'Taiwan'],
    // 日本地名與景點
    ['東京鐵塔', '東京タワー', 'Tokyo Tower'], ['晴空塔', '東京スカイツリー', 'Tokyo Skytree'],
    ['迪士尼樂園', 'ディズニーランド', 'Disneyland'], ['環球影城', 'ユニバーサル・スタジオ・ジャパン', 'Universal Studios Japan'],
    ['成田機場', '成田空港', 'Narita Airport'], ['羽田機場', '羽田空港', 'Haneda Airport'], ['關西機場', '関西空港', 'Kansai Airport'],
    ['淺草寺', '浅草寺', 'Senso-ji'], ['清水寺', '清水寺', 'Kiyomizu-dera'], ['道頓堀', '道頓堀', 'Dotonbori'],
    ['富士山', '富士山', 'Mount Fuji'], ['北海道', '北海道', 'Hokkaido'], ['秋葉原', '秋葉原', 'Akihabara'],
    ['東京', '東京', 'Tokyo'], ['大阪', '大阪', 'Osaka'], ['京都', '京都', 'Kyoto'], ['沖繩', '沖縄', 'Okinawa'],
    ['福岡', '福岡', 'Fukuoka'], ['名古屋', '名古屋', 'Nagoya'], ['札幌', '札幌', 'Sapporo'], ['奈良', '奈良', 'Nara'],
    ['橫濱', '横浜', 'Yokohama'], ['神戶', '神戸', 'Kobe'], ['新宿', '新宿', 'Shinjuku'], ['澀谷', '渋谷', 'Shibuya'],
    ['淺草', '浅草', 'Asakusa'], ['銀座', '銀座', 'Ginza'], ['上野', '上野', 'Ueno'], ['池袋', '池袋', 'Ikebukuro'],
    ['日本', '日本', 'Japan'],
    // 飲食
    ['牛肉麵', '牛肉麺', 'beef noodle soup'], ['珍珠奶茶', 'タピオカミルクティー', 'bubble tea'],
    ['小籠包', '小籠包', 'xiaolongbao'], ['滷肉飯', 'ルーローハン', 'braised pork rice'],
    ['鳳梨酥', 'パイナップルケーキ', 'pineapple cake'], ['臭豆腐', '臭豆腐', 'stinky tofu'],
    ['拉麵', 'ラーメン', 'ramen'], ['壽司', '寿司', 'sushi'], ['生魚片', '刺身', 'sashimi'], ['天婦羅', '天ぷら', 'tempura'],
    ['烏龍麵', 'うどん', 'udon'], ['蕎麥麵', 'そば', 'soba noodles'], ['章魚燒', 'たこ焼き', 'takoyaki'],
    ['大阪燒', 'お好み焼き', 'okonomiyaki'], ['居酒屋', '居酒屋', 'izakaya'], ['抹茶', '抹茶', 'matcha'],
    ['便當', '弁当', 'bento'], ['飯糰', 'おにぎり', 'onigiri'], ['味噌湯', '味噌汁', 'miso soup'], ['和牛', '和牛', 'wagyu'],
    ['日本酒', '日本酒', 'Japanese sake'], ['清酒', '日本酒', 'Japanese sake'],
    // 其他
    ['溫泉', '温泉', 'hot spring'], ['神社', '神社', 'shrine'], ['免稅', '免税', 'tax-free'],
].map(([zh, ja, en]) => ({ zh, ja, en }));

const escapeRegExp = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** 把句子裡的已知詞彙換成代號。回傳 { text, used: [{ placeholder, target }] } */
function applyGlossary(line, from, to, custom) {
    // 自訂優先；同一來源詞只留第一筆；比對時長的詞先比，避免「台北」搶走「台北車站」
    const seen = new Set();
    const entries = [...custom, ...BUILTIN_GLOSSARY]
        .filter((e) => e[from] && e[to] && !seen.has(e[from].toLowerCase()) && seen.add(e[from].toLowerCase()))
        .sort((a, b) => b[from].length - a[from].length);
    let text = line;
    const used = [];
    for (const entry of entries) {
        if (used.length >= PLACEHOLDERS.length) break;
        const placeholder = PLACEHOLDERS[used.length];
        const source = entry[from];
        // 英文要整個字才算（避免 Nara 比中 Narabi），中日文沒有空白，直接比對
        const pattern = from === 'en'
            ? new RegExp(`(?<![A-Za-z])${escapeRegExp(source)}(?![A-Za-z])`, 'gi')
            : new RegExp(escapeRegExp(source), 'g');
        if (!pattern.test(text)) continue;
        pattern.lastIndex = 0;
        text = text.replace(pattern, placeholder);
        used.push({ placeholder, target: entry[to] });
    }
    return { text, used };
}

/** 把代號換回譯名。有任何代號沒通過模型就回傳 null，由呼叫端改用一般翻譯 */
function restoreGlossary(text, used, to) {
    let out = text;
    for (const { placeholder, target } of used) {
        if (!new RegExp(placeholder, 'i').test(out)) return null;
        // 模型偶爾會把代號寫成「QZX(QZX)」，先收成一個
        out = out.replace(new RegExp(`(${placeholder})(\\s*[（(]\\s*${placeholder}\\s*[）)])+`, 'gi'), '$1');
        // 中日文的譯名前後不留空白；英文保留原本的空白
        out = to === 'en'
            ? out.replace(new RegExp(placeholder, 'gi'), target)
            : out.replace(new RegExp(`\\s*${placeholder}\\s*`, 'gi'), target);
    }
    return fixPunctuation(out, to);
}

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
// 記憶體充裕的裝置可同時留兩個主要模型（預覽用 base、最終結果用 small），不必來回重新載入。
const MAX_MAIN_ASR = (self.navigator?.deviceMemory ?? 4) >= 8 ? 2 : 1;
const asrPipes = new Map();
function getAsr(key, onProgress) {
    if (asrPipes.has(key)) {
        const p = asrPipes.get(key);   // 移到最後，表示最近用過
        asrPipes.delete(key);
        asrPipes.set(key, p);
        return p;
    }
    {
        if (key !== 'asr-live') {
            const mains = [...asrPipes.keys()].filter((k) => k !== 'asr-live');
            while (mains.length >= MAX_MAIN_ASR) {
                const oldest = mains.shift();
                asrPipes.get(oldest).then((p) => p.dispose?.()).catch(() => {});
                asrPipes.delete(oldest);
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
        if (target !== 'en' && !/[。！？…」』）?!.]$/.test(result)) result += '。';
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

async function translate(text, from, to, glossary = []) {
    const keys = route(from, to);
    const runChain = async (input) => {
        let current = input;
        for (const key of keys) {
            if (!current) break;
            current = await runStep(key, current);
        }
        return current;
    };
    const lines = [];
    for (const rawLine of text.split('\n')) {
        // 「臺」「台」統一成「台」再比對詞彙（語音辨識的輸出會用「臺」），不影響翻譯
        const line = from === 'zh' ? rawLine.trim().replace(/臺/g, '台') : rawLine.trim();
        if (!line) { lines.push(''); continue; }
        // 逐句套用詞彙：某一句的代號沒通過模型時，只有那一句退回一般翻譯，不拖累其他句
        const parts = [];
        for (const sentence of splitSentences(line)) {
            const { text: masked, used } = applyGlossary(sentence, from, to, glossary);
            let result = used.length ? restoreGlossary(await runChain(masked), used, to) : null;
            if (result === null) result = await runChain(sentence);
            if (result) parts.push(result);
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
            self.postMessage({ id, type: 'done', result: await translate(data.text, data.from, data.to, data.glossary ?? []) });
        }
    } catch (err) {
        self.postMessage({ id, type: 'error', message: String(err?.message ?? err) });
    }
}
