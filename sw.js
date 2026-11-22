// 譯機 service worker
// 改版時請調整 VERSION，舊快取會在 activate 時清掉。
const VERSION = 'v1';
const SHELL_CACHE = `tm-shell-${VERSION}`;
const CDN_CACHE = `tm-cdn-${VERSION}`;
const SHELL = [
    './',
    './index.html',
    './manifest.webmanifest',
    './icons/icon.svg',
    './icons/icon-192.png',
    './icons/icon-512.png',
    './icons/icon-maskable-512.png',
    './icons/apple-touch-icon.png',
];
// SDK 來源（版本固定，內容不會變）
const CDN_HOSTS = new Set(['esm.sh']);

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting())
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys()
            .then((keys) => Promise.all(
                keys.filter((k) => k.startsWith('tm-') && k !== SHELL_CACHE && k !== CDN_CACHE).map((k) => caches.delete(k))
            ))
            .then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', (event) => {
    const { request } = event;
    if (request.method !== 'GET') return;   // API 呼叫（POST）一律不經過快取
    const url = new URL(request.url);

    if (url.origin === self.location.origin) {
        event.respondWith(networkFirst(request));
    } else if (CDN_HOSTS.has(url.hostname)) {
        event.respondWith(cacheFirst(request));
    }
    // 其他來源（例如 api.anthropic.com）交給瀏覽器直接處理
});

// App 本體：有網路就拿最新版並更新快取，離線時退回快取
async function networkFirst(request) {
    const cache = await caches.open(SHELL_CACHE);
    try {
        const response = await fetch(request);
        if (response.ok) cache.put(request, response.clone());
        return response;
    } catch (err) {
        const cached = await cache.match(request, { ignoreSearch: true });
        if (cached) return cached;
        if (request.mode === 'navigate') {
            const shell = await cache.match('./index.html');
            if (shell) return shell;
        }
        throw err;
    }
}

// CDN 模組：快取優先
async function cacheFirst(request) {
    const cache = await caches.open(CDN_CACHE);
    const cached = await cache.match(request);
    if (cached) return cached;
    const response = await fetch(request);
    if (response.ok) cache.put(request, response.clone());
    return response;
}
