const PORT = Number(process.env.PORT ?? 8000);

const STREAM_REGISTRY_URL =
    process.env.STREAM_REGISTRY_URL ?? "http://localhost:8003";
const HLS_BASE_URL = process.env.HLS_BASE_URL ?? "http://localhost:8888";
const CAPTIONS_WS_BASE =
    process.env.CAPTIONS_WS_BASE ??
    process.env.WS_CAPTIONS_URL ??
    "ws://localhost:8002";

const textFile = (path, contentType) =>
    new Response(Bun.file(path), {
        headers: {
            "Content-Type": contentType,
            "Cache-Control": "no-store",
        },
    });

function escapeHtml(input = "") {
    return String(input)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function encodePathPreserveSlashes(path) {
    return String(path).split("/").map(encodeURIComponent).join("/");
}

function streamDisplayName(path) {
    return String(path).replace(/^live\//, "");
}

function pad2(value) {
    return String(value).padStart(2, "0");
}

function formatDateTimeRu(value) {
    if (!value) return "—";

    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "—";

    return `${pad2(date.getDate())}-${pad2(date.getMonth() + 1)}-${date.getFullYear()} ${pad2(date.getHours())}:${pad2(date.getMinutes())}:${pad2(date.getSeconds())}`;
}

function renderLayout({ title, body, page, streamPath = "" }) {
    return `<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1.0" />
  <meta name="color-scheme" content="dark light">
  <title>${escapeHtml(title)}</title>
  <link rel="stylesheet" href="/styles.css" />
  <script src="https://cdn.jsdelivr.net/npm/hls.js@1.5.18"></script>
</head>
<body data-page="${escapeHtml(page)}" data-stream-path="${escapeHtml(streamPath)}">
  <script>
    window.__APP_CONFIG__ = {
      apiBase: ${JSON.stringify(STREAM_REGISTRY_URL)},
      hlsBaseUrl: ${JSON.stringify(HLS_BASE_URL)},
      captionsWsBase: ${JSON.stringify(CAPTIONS_WS_BASE)}
    };
  </script>
  ${body}
  <script src="/app.js" defer></script>
  <script src="/favorite.js" defer></script>
</body>
</html>`;
}

function renderIndexPage() {
    return renderLayout({
        title: "Прямой эфир",
        page: "index",
        body: `
      <main class="app-shell">
        <header class="topbar panel">
          <div>
            <p class="eyebrow">браузер доступных трансляций</p>
            <h1>Прямой эфир</h1>
            <p class="muted">Список активных стримов</p>
          </div>
          <div class="topbar-actions">
            <label class="theme-control" for="theme-select">
              <button id="theme-switch" type="button" aria-label="Переключить тему" aria-pressed="false">
                <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="36px">
                  <path d="M480-120q-150 0-255-105T120-480q0-150 105-255t255-105q14 0 27.5 1t26.5 3q-41 29-65.5 75.5T444-660q0 90 63 153t153 63q55 0 101-24.5t75-65.5q2 13 3 26.5t1 27.5q0 150-105 255T480-120Z"/>
                </svg>

                <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="36px">
                  <path d="M480-280q-83 0-141.5-58.5T280-480q0-83 58.5-141.5T480-680q83 0 141.5 58.5T680-480q0 83-58.5 141.5T480-280ZM200-440H40v-80h160v80Zm720 0H760v-80h160v80ZM440-760v-160h80v160h-80Zm0 720v-160h80v160h-80ZM256-650l-101-97 57-59 96 100-52 56Zm492 496-97-101 53-55 101 97-57 59Zm-98-550 97-101 59 57-100 96-56-52ZM154-212l101-97 55 53-97 101-59-57Z"/>
                </svg>
              </button>
            </label>
          </div>
        </header>

        <section class="panel controls">
          <input id="search" class="field" type="search" placeholder="Поиск стрима..." />
          <button id="reload" class="button" type="button">Обновить</button>
        </section>

        <section class="panel list-panel">
          <div id="streams" class="stream-grid"></div>
          <div id="empty" class="empty hidden">Стримов пока нет</div>
          <div id="error" class="error hidden"></div>
        </section>
      </main>
    `,
    });
}

async function fetchStreams() {
    const url = new URL("/streams", STREAM_REGISTRY_URL);
    const res = await fetch(url);

    if (!res.ok) {
        throw new Error(
            `Stream registry error: ${res.status} ${res.statusText}`,
        );
    }

    const data = await res.json();
    const items = Array.isArray(data.items) ? data.items : [];

    return items
        .map((item) => {
            const startedAt =
                item.created_at ??
                item.createdAt ??
                item.updated_at ??
                item.updatedAt ??
                "";

            const displayName =
                item.display_name ??
                item.displayName ??
                streamDisplayName(item.path);

            return {
                path: item.path,
                name: displayName,
                displayName,
                sourceType: item.source_type ?? item.sourceType ?? "",
                sourceId: item.source_id ?? item.sourceId ?? "",
                rtspUrl: item.rtsp_url ?? item.rtspUrl ?? "",
                created: startedAt,
                createdAt: startedAt,
                startedAt,
                startedAtLabel: formatDateTimeRu(startedAt),
                updatedAt: item.updated_at ?? item.updatedAt ?? "",
                status: item.status ?? "",
            };
        })
        .sort((a, b) => a.path.localeCompare(b.path));
}

async function fetchStreamByPath(streamPath) {
    const streams = await fetchStreams();
    return streams.find((s) => s.path === streamPath) ?? null;
}

async function renderStreamPage(streamPath) {
    const stream = await fetchStreamByPath(streamPath);
    const name = stream?.displayName ?? streamDisplayName(streamPath);
    const startedAtLabel = stream?.startedAtLabel ?? "—";

    return renderLayout({
        title: `Stream: ${name}`,
        page: "stream",
        streamPath,
        body: `
      <main class="app-shell stream-shell">
        <header class="topbar panel">
          <div>
            <a class="backlink" href="/">← Назад</a>
            <h1>${escapeHtml(name)}</h1>
            <p class="muted">${escapeHtml(streamPath)}</p>
            <p class="muted">Запущен: ${escapeHtml(startedAtLabel)}</p>
          </div>
          <div class="topbar-actions">
            <label class="theme-control" for="theme-select">
              <button id="theme-switch" type="button" aria-label="Переключить тему" aria-pressed="false">
                <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="36px">
                  <path d="M480-120q-150 0-255-105T120-480q0-150 105-255t255-105q14 0 27.5 1t26.5 3q-41 29-65.5 75.5T444-660q0 90 63 153t153 63q55 0 101-24.5t75-65.5q2 13 3 26.5t1 27.5q0 150-105 255T480-120Z"/>
                </svg>

                <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="36px">
                  <path d="M480-280q-83 0-141.5-58.5T280-480q0-83 58.5-141.5T480-680q83 0 141.5 58.5T680-480q0 83-58.5 141.5T480-280ZM200-440H40v-80h160v80Zm720 0H760v-80h160v80ZM440-760v-160h80v160h-80Zm0 720v-160h80v160h-80ZM256-650l-101-97 57-59 96 100-52 56Zm492 496-97-101 53-55 101 97-57 59Zm-98-550 97-101 59 57-100 96-56-52ZM154-212l101-97 55 53-97 101-59-57Z"/>
                </svg>
              </button>
            </label>
          </div>
        </header>

        <section class="stream-layout">
          <article class="panel player-panel">
            <div class="video-shell" id="video-shell">
              <video
                id="video"
                class="video"
                controls
                autoplay
                playsinline
                muted
                preload="auto"
                controlsList="nodownload noplaybackrate noremoteplayback"
                disablePictureInPicture
              ></video>
            </div>

            <div class="section-head">
              <div>
                <h2>Статистика плеера</h2>
              </div>
              <span id="ws-status" class="chip">connecting</span>
            </div>

            <table class="stats-table" id="player-stats">
              <thead>
                <tr>
                  <th>Параметр</th>
                  <th>Значение</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </article>

          <aside class="panel captions-panel">
            <div class="section-head">
              <div>
                <h2>История</h2>
                <p class="muted">Последние цельные фразы</p>
              </div>
            </div>
            <div id="captions" class="cue-list"></div>
          </aside>
        </section>
      </main>
    `,
    });
}

Bun.serve({
    port: PORT,
    async fetch(req) {
        const url = new URL(req.url);

        if (url.pathname === "/") {
            return new Response(renderIndexPage(), {
                headers: { "Content-Type": "text/html; charset=utf-8" },
            });
        }

        if (url.pathname === "/styles.css") {
            return textFile("./public/styles.css", "text/css; charset=utf-8");
        }

        if (url.pathname === "/app.js") {
            return textFile(
                "./public/app.js",
                "application/javascript; charset=utf-8",
            );
        }

        if (url.pathname === "/favorite.js") {
            return textFile(
                "./public/favorite.js",
                "application/javascript; charset=utf-8",
            );
        }

        if (url.pathname === "/api/streams") {
            try {
                const streams = await fetchStreams();
                return Response.json({ items: streams });
            } catch (err) {
                return Response.json(
                    { error: String(err?.message ?? err) },
                    { status: 502 },
                );
            }
        }

        if (url.pathname.startsWith("/streams/")) {
            const streamPath = decodeURIComponent(
                url.pathname.slice("/streams/".length),
            );
            if (!streamPath) {
                return new Response("Not found", { status: 404 });
            }
            return new Response(await renderStreamPage(streamPath), {
                headers: { "Content-Type": "text/html; charset=utf-8" },
            });
        }

        return new Response("Not found", { status: 404 });
    },
});

console.log(`UI started on http://localhost:${PORT}`);
