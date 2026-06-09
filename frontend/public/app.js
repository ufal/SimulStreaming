const cfg = window.__APP_CONFIG__ || {};

function qs(sel) {
    return document.querySelector(sel);
}

const THEME_STORAGE_KEY = "stream-ui-theme";
const THEME_VALUES = new Set(["light", "dark"]);

function getStoredThemePreference() {
    try {
        const value = localStorage.getItem(THEME_STORAGE_KEY);
        return THEME_VALUES.has(value) ? value : "light";
    } catch {
        return "light";
    }
}

function applyTheme(theme) {
    const resolvedTheme = THEME_VALUES.has(theme) ? theme : "light";
    const root = document.documentElement;

    root.dataset.theme = resolvedTheme;
    root.style.colorScheme = resolvedTheme;

    return resolvedTheme;
}

function initThemeControls() {
    const btn = qs("#theme-switch");
    let theme = getStoredThemePreference();

    const sync = () => {
        theme = applyTheme(theme);

        if (btn) {
            btn.dataset.theme = theme;
            btn.setAttribute(
                "aria-pressed",
                theme === "dark" ? "true" : "false",
            );
        }
    };

    sync();

    btn?.addEventListener("click", () => {
        theme = theme === "dark" ? "light" : "dark";

        try {
            localStorage.setItem(THEME_STORAGE_KEY, theme);
        } catch {}

        sync();
    });
}

function encodePathPreserveSlashes(path) {
    return String(path).split("/").map(encodeURIComponent).join("/");
}

function escapeHtml(input = "") {
    return String(input)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function fmtDate(value) {
    if (!value) return "—";
    try {
        return new Date(value).toLocaleString();
    } catch {
        return String(value);
    }
}

function fmtKbps(value, fallback = "—") {
    if (value == null || Number.isNaN(Number(value))) return fallback;
    return `${Number(value).toFixed(0)} Kbps`;
}

function fmtSeconds(value, fallback = "—") {
    if (value == null || Number.isNaN(Number(value))) return fallback;
    return `${Number(value).toFixed(2)} сек.`;
}

function fmtFps(value, fallback = "—") {
    if (value == null || Number.isNaN(Number(value))) return fallback;
    const n = Number(value);
    return n >= 100 ? n.toFixed(0) : n.toFixed(1);
}

function formatSize(w, h) {
    if (!w || !h) return "—";
    return `${Math.round(w)}x${Math.round(h)}`;
}

function safeJsonParse(raw) {
    if (typeof raw !== "string") return raw;
    try {
        return JSON.parse(raw);
    } catch {
        return null;
    }
}

function normalizeText(input = "") {
    return String(input)
        .replace(/\u00a0/g, " ")
        .replace(/\r/g, "")
        .replace(/\n+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function normalizeCaptionFragment(input = "") {
    return String(input)
        .replace(/\u00a0/g, " ")
        .replace(/\r/g, "")
        .replace(/\n+/g, " ")
        .replace(/\s+/g, " ");
}

function formatClockTime(value) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    try {
        return new Date(Number(value)).toLocaleTimeString([], {
            hour12: false,
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        });
    } catch {
        return "—";
    }
}

function parseVttTimestamp(value) {
    const raw = String(value ?? "").trim();
    if (!raw) return null;

    const parts = raw.split(":");
    if (parts.length < 2) return null;

    const secBits = parts.pop().split(".");
    const seconds = Number(secBits[0]);
    const millis = Number((secBits[1] || "0").padEnd(3, "0"));
    const minutes = Number(parts.pop());
    const hours = Number(parts.length ? parts.pop() : "0");

    if (
        Number.isNaN(hours) ||
        Number.isNaN(minutes) ||
        Number.isNaN(seconds) ||
        Number.isNaN(millis)
    ) {
        return null;
    }

    return hours * 3600 + minutes * 60 + seconds + millis / 1000;
}

function formatVttTime(seconds) {
    if (seconds == null || Number.isNaN(Number(seconds))) {
        return new Date().toLocaleTimeString();
    }

    const total = Math.max(0, Number(seconds));
    const hh = Math.floor(total / 3600);
    const mm = Math.floor((total % 3600) / 60);
    const ss = Math.floor(total % 60);
    const ms = Math.floor((total - Math.floor(total)) * 1000);

    if (hh > 0) {
        return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
    }

    return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}

function parseWebVttDocument(rawText) {
    const raw = String(rawText ?? "")
        .replace(/\uFEFF/g, "")
        .replace(/\r/g, "");
    if (!raw.trim()) return [];

    const lines = raw.split("\n");
    let i = 0;
    const cues = [];

    if (lines[0]?.startsWith("WEBVTT")) {
        i += 1;
    }

    while (i < lines.length) {
        while (i < lines.length && lines[i].trim() === "") i += 1;
        if (i >= lines.length) break;

        if (lines[i].startsWith("NOTE")) {
            while (i < lines.length && lines[i].trim() !== "") i += 1;
            continue;
        }

        let cueId = "";
        let timingLine = lines[i];

        if (
            !timingLine.includes("-->") &&
            i + 1 < lines.length &&
            lines[i + 1].includes("-->")
        ) {
            cueId = timingLine;
            i += 1;
            timingLine = lines[i];
        }

        if (!timingLine || !timingLine.includes("-->")) {
            i += 1;
            continue;
        }

        const timingMatch = timingLine.match(
            /^\s*([0-9:\.]+)\s*-->\s*([0-9:\.]+)(.*)?$/,
        );

        let start = null;
        let end = null;

        if (timingMatch) {
            start = parseVttTimestamp(timingMatch[1]);
            end = parseVttTimestamp(timingMatch[2]);
        }

        i += 1;
        const textLines = [];

        while (i < lines.length && lines[i].trim() !== "") {
            textLines.push(lines[i]);
            i += 1;
        }

        const text = normalizeText(textLines.join(" "));
        if (text && text !== "WEBVTT") {
            cues.push({ cueId, start, end, text });
        }
    }

    return cues;
}

function extractCaptionEvents(payload) {
    if (!payload) return [];

    const events = [];
    const seen = new Set();

    const addEvent = (event) => {
        const text = normalizeText(event?.text ?? "");
        if (!text) return;

        const key = [
            event?.cueId ?? "",
            event?.start ?? "",
            event?.end ?? "",
            text,
        ].join("|");
        if (seen.has(key)) return;
        seen.add(key);

        events.push({
            text,
            cueId: event?.cueId ?? "",
            start: event?.start ?? null,
            end: event?.end ?? null,
            sourceCaptureUnixMS:
                event?.sourceCaptureUnixMS == null ||
                Number.isNaN(Number(event?.sourceCaptureUnixMS))
                    ? null
                    : Number(event.sourceCaptureUnixMS),
            latencyMs:
                event?.latencyMs == null ||
                Number.isNaN(Number(event?.latencyMs))
                    ? null
                    : Number(event.latencyMs),
        });
    };

    const pushFromString = (text) => {
        const raw = String(text ?? "");
        if (!raw.trim()) return;

        if (raw.trimStart().startsWith("WEBVTT") || raw.includes("-->")) {
            const cues = parseWebVttDocument(raw);
            if (cues.length) {
                for (const cue of cues) addEvent(cue);
                return;
            }
        }

        addEvent({ text: raw });
    };

    if (typeof payload === "string") {
        pushFromString(payload);
        return events;
    }

    if (Array.isArray(payload)) {
        for (const item of payload) {
            const nested = extractCaptionEvents(item);
            for (const ev of nested) addEvent(ev);
        }
        return events;
    }

    if (typeof payload !== "object") {
        return events;
    }

    const merged =
        payload?.payload && typeof payload.payload === "object"
            ? { ...payload, ...payload.payload }
            : payload;

    const candidates = [
        merged?.text,
        merged?.caption,
        merged?.body,
        merged?.raw,
        merged?.vtt,
        merged?.webvtt,
        merged?.message?.text,
        merged?.message?.body,
        merged?.data?.text,
        merged?.data?.body,
    ];

    const start =
        merged?.start ?? merged?.cue?.start ?? merged?.payload?.start ?? null;
    const end = merged?.end ?? merged?.cue?.end ?? merged?.payload?.end ?? null;
    const cueId =
        merged?.cue_id ??
        merged?.cueId ??
        merged?.payload?.cue_id ??
        merged?.payload?.cueId ??
        "";
    const sourceCaptureUnixMS =
        merged?.source_capture_unix_ms ??
        merged?.sourceCaptureUnixMS ??
        merged?.payload?.source_capture_unix_ms ??
        merged?.payload?.sourceCaptureUnixMS ??
        null;
    const latencyMs =
        merged?.latency_ms ??
        merged?.latencyMs ??
        merged?.payload?.latency_ms ??
        merged?.payload?.latencyMs ??
        null;

    for (const candidate of candidates) {
        if (typeof candidate === "string" && candidate.trim()) {
            if (
                candidate.trimStart().startsWith("WEBVTT") ||
                candidate.includes("-->")
            ) {
                const cues = parseWebVttDocument(candidate);
                if (cues.length) {
                    for (const cue of cues) addEvent(cue);
                    return events;
                }
            }

            addEvent({
                text: candidate,
                cueId,
                start,
                end,
                sourceCaptureUnixMS,
                latencyMs,
            });
            return events;
        }
    }

    return events;
}

function getBufferedAhead(video) {
    try {
        if (!video || !video.buffered || video.buffered.length === 0) return 0;
        const t = video.currentTime || 0;

        for (let i = 0; i < video.buffered.length; i += 1) {
            const start = video.buffered.start(i);
            const end = video.buffered.end(i);
            if (t >= start && t <= end) {
                return Math.max(0, end - t);
            }
        }

        return 0;
    } catch {
        return 0;
    }
}

function getLiveLatency(video) {
    try {
        if (!video || !video.seekable || video.seekable.length === 0)
            return null;
        const liveEdge = video.seekable.end(video.seekable.length - 1);
        if (!Number.isFinite(liveEdge)) return null;
        return Math.max(0, liveEdge - (video.currentTime || 0));
    } catch {
        return null;
    }
}

function getVideoQuality(video) {
    if (!video) return { total: 0, dropped: 0 };
    const q = video.getVideoPlaybackQuality?.();
    const dropped = q?.droppedVideoFrames ?? video.webkitDroppedFrameCount ?? 0;
    const total = q?.totalVideoFrames ?? video.webkitDecodedFrameCount ?? 0;
    return { total, dropped };
}

function collectCodecs(level) {
    const combined =
        level?.codecs ??
        level?.attrs?.CODECS ??
        [level?.videoCodec, level?.audioCodec].filter(Boolean).join(",");

    if (!combined) return "";
    return String(combined)
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
        .filter((v, i, arr) => arr.indexOf(v) === i)
        .join(", ");
}

function getCurrentLevel(hls) {
    if (!hls?.levels?.length) return null;
    const idx =
        hls.currentLevel >= 0
            ? hls.currentLevel
            : hls.nextLevel >= 0
              ? hls.nextLevel
              : 0;
    return hls.levels[idx] ?? hls.levels[0] ?? null;
}

function formatLatencyMs(value, digits = 1) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    return `${Number(value).toFixed(digits)} ms`;
}

function percentile(sortedValues, p) {
    if (!sortedValues.length) return null;
    if (sortedValues.length === 1) return sortedValues[0];

    const index = (sortedValues.length - 1) * p;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);

    if (lower === upper) return sortedValues[lower];

    const weight = index - lower;
    return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}

function calculateLatencyStats(samples) {
    const values = (Array.isArray(samples) ? samples : [])
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value) && value > 0)
        .slice()
        .sort((a, b) => a - b);

    if (!values.length) return null;

    const sum = values.reduce((acc, value) => acc + value, 0);

    return {
        count: values.length,
        avg: sum / values.length,
        min: values[0],
        max: values[values.length - 1],
        p50: percentile(values, 0.5),
        p75: percentile(values, 0.75),
        p95: percentile(values, 0.95),
        p99: percentile(values, 0.99),
    };
}

function buildStatsRows({ hls, video, frame, fps }) {
    const state = ensureState();
    const level = getCurrentLevel(hls);

    const loadedW =
        level?.width ?? level?.attrs?.RESOLUTION?.split("x")?.[0] ?? null;
    const loadedH =
        level?.height ?? level?.attrs?.RESOLUTION?.split("x")?.[1] ?? null;

    const processedW = video?.videoWidth ?? null;
    const processedH = video?.videoHeight ?? null;

    const rect = frame?.getBoundingClientRect?.() ?? null;
    const viewportW = rect ? Math.round(rect.width) : null;
    const viewportH = rect ? Math.round(rect.height) : null;

    const bitrate = level?.bitrate ?? null;
    const bwEstimate = hls?.bandwidthEstimate ?? null;
    const fpsValue = fps ?? level?.frameRate ?? level?.fps ?? null;

    const { dropped } = getVideoQuality(video);
    const bufferLen = getBufferedAhead(video);
    const latency = getLiveLatency(video);
    const codecs = collectCodecs(level);

    const latencyStats = calculateLatencyStats(
        (state.captionsStat || [])
            .map((item) => Number(item?.latencyMs))
            .filter((value) => Number.isFinite(value) && value > 0),
    );

    return [
        { type: "section", label: "Текущие HLS/player stats" },
        {
            type: "row",
            name: "Разрешение загрузки",
            value: formatSize(loadedW, loadedH),
        },
        {
            type: "row",
            name: "Разрешение обработки",
            value: formatSize(processedW, processedH),
        },
        {
            type: "row",
            name: "Разрешение области просмотра",
            value: formatSize(viewportW, viewportH),
        },
        {
            type: "row",
            name: "Битрейт загрузки",
            value: bitrate ? fmtKbps(bitrate / 1000) : "—",
        },
        {
            type: "row",
            name: "Оценка пропускной способности",
            value: bwEstimate ? fmtKbps(bwEstimate / 1000) : "—",
        },
        { type: "row", name: "Кадров в секунду", value: fmtFps(fpsValue) },
        { type: "row", name: "Пропущенные кадры", value: String(dropped ?? 0) },
        { type: "row", name: "Размер буфера", value: fmtSeconds(bufferLen) },
        {
            type: "row",
            name: "Задержка до live edge",
            value: latency == null ? "—" : fmtSeconds(latency),
        },
        { type: "row", name: "Кодеки", value: codecs || "—" },
        { type: "row", name: "Протокол", value: "HLS" },
        {
            type: "row",
            name: "Режим задержки",
            value: hls?.config?.lowLatencyMode ? "Низкая задержка" : "Обычный",
        },
        {
            type: "section",
            label: "Сквозная задержка субтитров (latency_ms_client)",
        },
        {
            type: "row",
            name: "avg",
            value: latencyStats ? formatLatencyMs(latencyStats.avg, 1) : "—",
        },
        {
            type: "row",
            name: "min",
            value: latencyStats ? `${Math.round(latencyStats.min)} ms` : "—",
        },
        {
            type: "row",
            name: "max",
            value: latencyStats ? `${Math.round(latencyStats.max)} ms` : "—",
        },
        {
            type: "row",
            name: "p50",
            value: latencyStats ? formatLatencyMs(latencyStats.p50, 1) : "—",
        },
        {
            type: "row",
            name: "p75",
            value: latencyStats ? formatLatencyMs(latencyStats.p75, 1) : "—",
        },
        {
            type: "row",
            name: "p95",
            value: latencyStats ? formatLatencyMs(latencyStats.p95, 1) : "—",
        },
        {
            type: "row",
            name: "p99",
            value: latencyStats ? formatLatencyMs(latencyStats.p99, 1) : "—",
        },
    ];
}

function setStatsTable(statsEl, rows) {
    if (!statsEl) return;
    const tbody = statsEl.querySelector("tbody");
    const headerCells = statsEl.querySelectorAll("thead th");
    if (headerCells[1]) {
        headerCells[1].style.textAlign = "right";
    }
    if (!tbody) return;

    tbody.innerHTML = rows
        .map((row) => {
            if (Array.isArray(row)) {
                const [name, value] = row;
                return `
        <tr>
          <td>${escapeHtml(name)}</td>
          <td>${escapeHtml(value)}</td>
        </tr>
      `;
            }

            if (row?.type === "section") {
                return `
        <tr class="stats-section-row">
          <td colspan="2" style="background:var(--table-head);font-weight:700;color:var(--text);text-align:left;border-top:1px solid var(--line-strong);padding:12px 14px;">
            ${escapeHtml(row.label)}
          </td>
        </tr>
      `;
            }

            const name = row?.name ?? "";
            const value = row?.value ?? "—";

            return `
        <tr>
          <td>${escapeHtml(name)}</td>
          <td>${escapeHtml(value)}</td>
        </tr>
      `;
        })
        .join("");
}

function ensureState() {
    if (!window.__STREAM_UI_STATE__) {
        window.__STREAM_UI_STATE__ = {
            historyItems: [],
            captionsStat: [],
            track: null,
            trackCues: [],
            autoAdjustSuspendedUntil: 0,
        };
    }
    return window.__STREAM_UI_STATE__;
}

function ensureUiInteractionState() {
    const state = ensureState();
    if (typeof state.autoAdjustSuspendedUntil !== "number") {
        state.autoAdjustSuspendedUntil = 0;
    }
    return state;
}

function suspendAutoAdjustments(ms = 5000) {
    const state = ensureUiInteractionState();
    state.autoAdjustSuspendedUntil = Math.max(
        state.autoAdjustSuspendedUntil,
        Date.now() + ms,
    );
}

function autoAdjustmentsAllowed() {
    const state = ensureUiInteractionState();
    return Date.now() >= state.autoAdjustSuspendedUntil;
}

function isAtBottom(box) {
    return box.scrollTop + box.clientHeight >= box.scrollHeight - 8;
}

function buildHistoryTimeLabel(item) {
    if (item.displayStartUnixMS != null && item.displayEndUnixMS != null) {
        return `${formatClockTime(item.displayStartUnixMS)} → ${formatClockTime(item.displayEndUnixMS)}`;
    }
    if (item.displayStartUnixMS != null) {
        return formatClockTime(item.displayStartUnixMS);
    }
    return new Date(item.arrival ?? Date.now()).toLocaleTimeString();
}

function renderHistory() {
    const box = qs("#captions");
    if (!box) return;

    const state = ensureState();
    const prevScrollTop = box.scrollTop;
    const stickToBottom =
        isAtBottom(box) || box.scrollHeight <= box.clientHeight + 2;

    box.innerHTML = state.historyItems
        .map(
            (item) => `
        <article class="cue-item">
          <div class="cue-time">${escapeHtml(buildHistoryTimeLabel(item))}</div>
          <div class="cue-text">${escapeHtml(item.text)}</div>
        </article>
      `,
        )
        .join("");

    if (!state.historyItems.length) {
        box.scrollTop = 0;
        return;
    }

    if (stickToBottom) {
        box.scrollTop = box.scrollHeight;
        return;
    }

    const maxScrollTop = Math.max(0, box.scrollHeight - box.clientHeight);
    box.scrollTop = Math.min(prevScrollTop, maxScrollTop);
}

function pushHistoryItem(item) {
    const state = ensureState();
    const text = normalizeText(item?.text ?? "");
    if (!text) return;

    const start = Number.isFinite(Number(item?.start))
        ? Number(item.start)
        : null;
    const end = Number.isFinite(Number(item?.end)) ? Number(item.end) : null;

    state.historyItems.push({
        text,
        start,
        end,
        displayStartUnixMS: Number.isFinite(Number(item?.displayStartUnixMS))
            ? Number(item.displayStartUnixMS)
            : null,
        displayEndUnixMS: Number.isFinite(Number(item?.displayEndUnixMS))
            ? Number(item.displayEndUnixMS)
            : null,
        arrival: item?.arrival ?? Date.now(),
    });

    if (state.historyItems.length > 500) {
        state.historyItems = state.historyItems.slice(-500);
    }

    renderHistory();
}

function addCaptionStat(sample) {
    const state = ensureState();

    const sourceCaptureUnixMS = Number.isFinite(
        Number(sample?.sourceCaptureUnixMS),
    )
        ? Number(sample.sourceCaptureUnixMS)
        : Number.isFinite(Number(sample?.source_capture_unix_ms))
          ? Number(sample.source_capture_unix_ms)
          : null;

    const latencyMs = Number.isFinite(Number(sample?.latencyMs))
        ? Number(sample.latencyMs)
        : sourceCaptureUnixMS != null
          ? Date.now() - sourceCaptureUnixMS
          : null;

    if (!Number.isFinite(latencyMs) || latencyMs <= 0) return false;

    state.captionsStat.push({
        text: normalizeText(sample?.text ?? ""),
        start: Number.isFinite(Number(sample?.start))
            ? Number(sample.start)
            : null,
        end: Number.isFinite(Number(sample?.end)) ? Number(sample.end) : null,
        arrival: sample?.arrival ?? Date.now(),
        sourceCaptureUnixMS,
        latencyMs,
    });

    if (state.captionsStat.length > 1000) {
        state.captionsStat = state.captionsStat.slice(-1000);
    }

    return true;
}

function isPunctuationOnly(token) {
    return /^[\p{P}\p{S}]+$/u.test(token);
}

function isDashLikePunctuation(token) {
    return /^[\u002D\u2010-\u2015\u2212]+$/u.test(token);
}

function extractWords(buffer) {
    const words = [];
    let i = 0;
    const n = buffer.length;

    while (i < n && /\s/u.test(buffer[i])) i += 1;

    let start = i;

    while (i < n) {
        if (/\s/u.test(buffer[i])) {
            if (start < i) words.push(buffer.slice(start, i));
            i += 1;
            while (i < n && /\s/u.test(buffer[i])) i += 1;
            start = i;
        } else {
            i += 1;
        }
    }

    const tail = start < n ? buffer.slice(start) : "";

    if (buffer && !/\s/u.test(buffer[buffer.length - 1])) {
        return [words, tail];
    }

    return [words, ""];
}

class SubtitleWindow {
    constructor(width = 60) {
        this.width = width;
        this.lineLen = 0;
        this.currentLine = "";
    }

    willOverflow(word) {
        if (!word) return false;

        const punct = isPunctuationOnly(word);
        const isDash = isDashLikePunctuation(word);
        const needsSpace = this.lineLen > 0 && (!punct || isDash);
        const needed =
            this.lineLen === 0
                ? word.length
                : this.lineLen + (needsSpace ? 1 : 0) + word.length;
        return this.lineLen > 0 && needed > this.width;
    }

    pushWord(word) {
        if (!word) return this.currentLine;

        const punct = isPunctuationOnly(word);
        const isDash = isDashLikePunctuation(word);
        const needsSpace = this.lineLen > 0 && (!punct || isDash);

        const needed =
            this.lineLen === 0
                ? word.length
                : this.lineLen + (needsSpace ? 1 : 0) + word.length;

        if (this.lineLen > 0 && needed > this.width) {
            this.currentLine = "";
            this.lineLen = 0;
        }

        if (this.lineLen > 0 && (!punct || isDash)) {
            this.currentLine += " ";
            this.lineLen += 1;
        }

        this.currentLine += word;
        this.lineLen += word.length;
        return this.currentLine;
    }
}

const WINDOW_SIZE = 60;
const MIN_CUE_DURATION = 0.07;
const MIN_STEP = 0.07;
const SYNC_PADDING_SEC = 0.02;
const CUE_STICKY_TAIL_SEC = 3;
const LINE_BREAK_DELAY_SEC = 0.3;
const WINDOW_GAP_RESET_SEC = 1.25;

class SubtitleRuntime {
    constructor(video, track, onStatsUpdated = null) {
        this.video = video;
        this.track = track;
        this.onStatsUpdated = onStatsUpdated;

        this.window = new SubtitleWindow(WINDOW_SIZE);
        this.textBuffer = "";
        this.pendingLeadingPunct = "";

        this.anchorCaptureUnixMs = null;
        this.anchorMediaTime = null;
        this.estimatedDelaySec = null;

        this.queue = [];
        this.ready = false;

        this.activeCue = null;
        this.windowStartTime = null;

        this.currentWindowStartTime = null;
        this.currentWindowLastWordTime = null;

        this.currentWindowStartCaptureUnixMS = null;
        this.currentWindowLastCaptureUnixMS = null;

        this.wordQueue = [];
        this.renderTimer = null;
    }

    setReady() {
        this.ready = true;
        while (this.queue.length > 0) {
            this.extractPayload(this.queue.shift());
        }
    }

    enqueue(payload) {
        if (this.ready) {
            this.extractPayload(payload);
        } else {
            this.queue.push(payload);
        }
    }

    updateDelayEstimate(captureUnixMs) {
        const now = Date.now();
        const sampleSec = Math.max(
            0,
            (now - captureUnixMs) / 1000 + SYNC_PADDING_SEC,
        );

        if (this.estimatedDelaySec === null) {
            this.estimatedDelaySec = sampleSec;
        } else {
            const alpha = 0.18;
            this.estimatedDelaySec =
                this.estimatedDelaySec * (1 - alpha) + sampleSec * alpha;
        }

        return this.estimatedDelaySec;
    }

    establishAnchor(captureUnixMs) {
        const delaySec = this.updateDelayEstimate(captureUnixMs);

        if (this.anchorCaptureUnixMs === null) {
            this.anchorCaptureUnixMs = captureUnixMs;
            this.anchorMediaTime = (this.video.currentTime || 0) + delaySec;
        }

        return this.anchorMediaTime;
    }

    captureToMediaTime(captureUnixMs) {
        if (
            this.anchorCaptureUnixMs === null ||
            this.anchorMediaTime === null
        ) {
            return this.video.currentTime || 0;
        }

        return (
            this.anchorMediaTime +
            (captureUnixMs - this.anchorCaptureUnixMs) / 1000
        );
    }

    makeTiming(payload, wordCount) {
        const startLocal = Number.isFinite(Number(payload?.start))
            ? Number(payload.start)
            : null;
        const endLocal = Number.isFinite(Number(payload?.end))
            ? Number(payload.end)
            : null;
        const segmentDuration =
            startLocal !== null && endLocal !== null
                ? Math.max(endLocal - startLocal, MIN_CUE_DURATION)
                : MIN_CUE_DURATION;

        const captureUnixMs = Number.isFinite(
            Number(payload?.source_capture_unix_ms),
        )
            ? Number(payload.source_capture_unix_ms)
            : Number.isFinite(Number(payload?.sourceCaptureUnixMS))
              ? Number(payload.sourceCaptureUnixMS)
              : null;

        const baseTime = this.captureToMediaTime(
            captureUnixMs !== null ? captureUnixMs : Date.now(),
        );

        const step = Math.max(
            segmentDuration / Math.max(wordCount, 1),
            MIN_STEP,
        );

        return { baseTime, step, segmentDuration, captureUnixMs };
    }

    ensureCue(startTime, endTime, text) {
        if (this.activeCue) {
            try {
                this.track.removeCue(this.activeCue);
            } catch {}
            this.activeCue = null;
        }

        const CueClass = window.VTTCue || window.TextTrackCue;
        if (typeof CueClass !== "function") return;

        const cue = new CueClass(startTime, endTime, text);
        cue.snapToLines = true;
        cue.line = -2;
        cue.align = "start";
        cue.position = 3;
        cue.size = 100;

        if ("positionAlign" in cue) {
            cue.positionAlign = "line-left";
        }

        try {
            this.track.addCue(cue);
            this.activeCue = cue;
        } catch (err) {
            console.error("cue add error", err);
        }
    }

    commitCurrentWindow() {
        const text = normalizeText(this.window.currentLine);
        const start = this.currentWindowStartTime ?? this.windowStartTime;
        const end = this.currentWindowLastWordTime ?? start;

        if (!text || start == null || end == null) return false;

        const displayStartUnixMS =
            this.currentWindowStartCaptureUnixMS ??
            this.anchorCaptureUnixMs ??
            Date.now();
        const displayEndUnixMS =
            this.currentWindowLastCaptureUnixMS ?? displayStartUnixMS;

        pushHistoryItem({
            text,
            start,
            end,
            displayStartUnixMS,
            displayEndUnixMS,
            arrival: Date.now(),
        });

        return true;
    }

    breakCurrentWindow(nextWordTime) {
        this.commitCurrentWindow();
        this.window.currentLine = "";
        this.window.lineLen = 0;
        this.pendingLeadingPunct = "";
        this.windowStartTime = nextWordTime;
        this.currentWindowStartTime = nextWordTime;
        this.currentWindowLastWordTime = null;
        this.currentWindowStartCaptureUnixMS = null;
        this.currentWindowLastCaptureUnixMS = null;
    }

    extractPayload(payload) {
        if (payload == null) return;

        const merged =
            payload?.payload && typeof payload.payload === "object"
                ? { ...payload, ...payload.payload }
                : payload;

        if (
            merged?.kind &&
            merged.kind !== "subtitle" &&
            merged.kind !== "caption" &&
            merged.kind !== "subtitles" &&
            !String(merged?.text ?? "").trim() &&
            !String(merged?.body ?? "").trim()
        ) {
            return;
        }

        const rawIncoming =
            merged?.text ??
            merged?.caption ??
            merged?.body ??
            merged?.raw ??
            merged?.vtt ??
            merged?.webvtt ??
            merged?.message?.text ??
            merged?.message?.body ??
            merged?.data?.text ??
            merged?.data?.body ??
            "";

        const incomingText = normalizeCaptionFragment(rawIncoming);

        if (!incomingText.trim()) return;

        const captureUnixMs = Number.isFinite(
            Number(merged?.source_capture_unix_ms),
        )
            ? Number(merged.source_capture_unix_ms)
            : Number.isFinite(Number(merged?.sourceCaptureUnixMS))
              ? Number(merged.sourceCaptureUnixMS)
              : null;

        if (captureUnixMs !== null) {
            this.establishAnchor(captureUnixMs);
        } else if (this.anchorCaptureUnixMs === null) {
            this.anchorCaptureUnixMs = Date.now();
            this.anchorMediaTime = this.video.currentTime || 0;
        }

        addCaptionStat({
            text: incomingText,
            start: merged?.start ?? null,
            end: merged?.end ?? null,
            sourceCaptureUnixMS: captureUnixMs,
            arrival: Date.now(),
        });

        if (typeof this.onStatsUpdated === "function") {
            this.onStatsUpdated();
        }

        if (
            incomingText.trimStart().startsWith("WEBVTT") ||
            incomingText.includes("-->")
        ) {
            const cues = parseWebVttDocument(incomingText);
            if (!cues.length) return;

            for (const cue of cues) {
                if (!cue.text) continue;

                const start = Number.isFinite(Number(cue.start))
                    ? Number(cue.start)
                    : this.video.currentTime || 0;
                const end = Number.isFinite(Number(cue.end))
                    ? Number(cue.end)
                    : start + 2;

                this.ensureCue(start, end, cue.text);
            }

            return;
        }

        this.textBuffer += incomingText;

        let [words, tail] = extractWords(this.textBuffer);
        this.textBuffer = tail;

        if (!words.length) return;

        const normalized = [];
        for (const token of words) {
            if (isPunctuationOnly(token)) {
                if (isDashLikePunctuation(token)) {
                    const withPrefix = this.pendingLeadingPunct
                        ? this.pendingLeadingPunct + token
                        : token;
                    this.pendingLeadingPunct = "";
                    normalized.push(withPrefix);
                } else {
                    if (normalized.length > 0) {
                        normalized[normalized.length - 1] += token;
                    } else {
                        this.pendingLeadingPunct += token;
                    }
                }
            } else {
                const withPrefix = this.pendingLeadingPunct
                    ? this.pendingLeadingPunct + token
                    : token;
                this.pendingLeadingPunct = "";
                normalized.push(withPrefix);
            }
        }

        if (!normalized.length) return;

        const {
            baseTime,
            step,
            segmentDuration,
            captureUnixMs: batchCaptureMs,
        } = this.makeTiming(merged, normalized.length);

        for (let i = 0; i < normalized.length; i += 1) {
            const wordTime = baseTime + i * step;
            const desiredEnd = Math.max(
                wordTime + MIN_CUE_DURATION,
                wordTime + step,
                baseTime + segmentDuration,
                wordTime + CUE_STICKY_TAIL_SEC,
            );

            this.wordQueue.push({
                word: normalized[i],
                wordTime,
                desiredEnd,
                captureUnixMs: batchCaptureMs,
            });
        }

        this.scheduleRender();
    }

    scheduleRender() {
        if (this.renderTimer) return;

        const tick = () => {
            if (this.wordQueue.length === 0) {
                this.renderTimer = null;
                return;
            }

            const item = this.wordQueue[0];
            const nextWindowShouldBreak =
                this.currentWindowLastWordTime != null &&
                item.wordTime - this.currentWindowLastWordTime >
                    WINDOW_GAP_RESET_SEC;
            const willReset = this.window.willOverflow(item.word);

            if (
                (nextWindowShouldBreak || willReset) &&
                this.window.lineLen > 0
            ) {
                this.breakCurrentWindow(item.wordTime);
            } else if (this.window.lineLen === 0) {
                this.windowStartTime = item.wordTime;
                this.currentWindowStartTime = item.wordTime;
            }

            const extraDelay = willReset ? LINE_BREAK_DELAY_SEC : 0;
            const targetTime = item.wordTime + extraDelay;
            const now = this.video.currentTime || 0;
            const isDesynced = Math.abs(targetTime - now) > 15;

            if (now >= targetTime || isDesynced) {
                this.wordQueue.shift();

                if (this.window.lineLen === 0 && this.windowStartTime == null) {
                    this.windowStartTime = item.wordTime;
                    this.currentWindowStartTime = item.wordTime;
                }

                const currentText = this.window.pushWord(item.word);

                if (this.currentWindowStartTime == null) {
                    this.currentWindowStartTime = item.wordTime;
                }
                this.currentWindowLastWordTime = item.wordTime;

                if (this.currentWindowStartCaptureUnixMS == null) {
                    this.currentWindowStartCaptureUnixMS =
                        item.captureUnixMs ?? Date.now();
                }
                this.currentWindowLastCaptureUnixMS =
                    item.captureUnixMs ?? this.currentWindowStartCaptureUnixMS;

                this.ensureCue(
                    this.currentWindowStartTime,
                    item.desiredEnd,
                    currentText,
                );

                this.renderTimer = setTimeout(tick, 0);
            } else {
                this.renderTimer = setTimeout(tick, 50);
            }
        };

        this.renderTimer = setTimeout(tick, 0);
    }
}

function waitForBuffer(video, targetSeconds, timeoutMs = 30000) {
    const startedAt = Date.now();

    return (async () => {
        while (Date.now() - startedAt < timeoutMs) {
            if (getBufferedAhead(video) >= targetSeconds) return true;
            await new Promise((r) => setTimeout(r, 150));
        }

        return false;
    })();
}

function createFpsMonitor(video, onUpdate) {
    let stopped = false;
    let fps = null;

    if (typeof video.requestVideoFrameCallback === "function") {
        let lastNow = null;
        let lastPresented = null;

        const tick = (now, metadata) => {
            if (stopped) return;

            const presented = metadata?.presentedFrames ?? null;

            if (lastNow == null) {
                lastNow = now;
                lastPresented = presented;
            } else {
                const elapsed = (now - lastNow) / 1000;
                const frames =
                    presented != null && lastPresented != null
                        ? presented - lastPresented
                        : null;

                if (elapsed >= 1) {
                    if (frames != null && frames >= 0) {
                        fps = frames / elapsed;
                        onUpdate();
                    }
                    lastNow = now;
                    lastPresented = presented;
                }
            }

            video.requestVideoFrameCallback(tick);
        };

        video.requestVideoFrameCallback(tick);

        return {
            get value() {
                return fps;
            },
            destroy() {
                stopped = true;
            },
        };
    }

    let lastWall = performance.now();
    let lastDecoded = 0;

    const timer = setInterval(() => {
        if (stopped) return;

        const q = video.getVideoPlaybackQuality?.();
        const decoded =
            q?.totalVideoFrames ?? video.webkitDecodedFrameCount ?? 0;
        const now = performance.now();
        const elapsed = (now - lastWall) / 1000;

        if (elapsed >= 1) {
            const frames = decoded - lastDecoded;
            if (Number.isFinite(frames) && frames >= 0) {
                fps = frames / elapsed;
                onUpdate();
            }
            lastWall = now;
            lastDecoded = decoded;
        }
    }, 500);

    return {
        get value() {
            return fps;
        },
        destroy() {
            stopped = true;
            clearInterval(timer);
        },
    };
}

function toggleFullscreen(frame) {
    const request = async () => {
        if (document.fullscreenElement === frame) return;
        if (frame.requestFullscreen) return frame.requestFullscreen();
        if (frame.webkitRequestFullscreen)
            return frame.webkitRequestFullscreen();
        if (frame.mozRequestFullScreen) return frame.mozRequestFullScreen();
        if (frame.msRequestFullscreen) return frame.msRequestFullscreen();
        throw new Error("Fullscreen API is not supported");
    };

    const exit = async () => {
        if (!document.fullscreenElement) return;
        if (document.exitFullscreen) return document.exitFullscreen();
        if (document.webkitExitFullscreen)
            return document.webkitExitFullscreen();
        if (document.mozCancelFullScreen) return document.mozCancelFullScreen();
        if (document.msExitFullscreen) return document.msExitFullscreen();
        throw new Error("Fullscreen API is not supported");
    };

    if (document.fullscreenElement === frame) {
        return exit();
    }

    return request();
}

function bindFullscreenControls(frame, video, statsEl, getHls) {
    const btn = qs("#fullscreen-toggle");

    const syncButtonState = () => {
        const isFs = document.fullscreenElement === frame;
        document.body.classList.toggle(
            "player-fullscreen",
            !!document.fullscreenElement,
        );
        frame.classList.toggle("is-fullscreen", isFs);

        if (btn) {
            btn.dataset.active = isFs ? "true" : "false";
            btn.textContent = isFs ? "Exit" : "Fullscreen";
        }
    };

    const fpsMonitor = createFpsMonitor(video, () => updateStats());

    const onToggle = async () => {
        try {
            await toggleFullscreen(frame);
            syncButtonState();
        } catch (e) {
            console.error("fullscreen error", e);
        }
    };

    if (btn) {
        btn.addEventListener("click", onToggle);
    }

    document.addEventListener("fullscreenchange", syncButtonState);
    document.addEventListener("webkitfullscreenchange", syncButtonState);
    syncButtonState();

    function updateStats() {
        setStatsTable(
            statsEl,
            buildStatsRows({
                hls: getHls(),
                video,
                frame,
                fps: fpsMonitor.value,
            }),
        );
    }

    return {
        updateStats,
        destroy() {
            fpsMonitor.destroy();
            if (btn) {
                btn.removeEventListener("click", onToggle);
            }
            document.removeEventListener("fullscreenchange", syncButtonState);
            document.removeEventListener(
                "webkitfullscreenchange",
                syncButtonState,
            );
        },
    };
}

async function loadStreams() {
    const container = qs("#streams");
    const empty = qs("#empty");
    const error = qs("#error");
    const search = qs("#search");
    let cachedItems = [];

    const render = (items) => {
        const query = (search?.value || "").trim().toLowerCase();
        const filtered = items.filter((s) => {
            if (!query) return true;
            return (
                s.path.toLowerCase().includes(query) ||
                s.name.toLowerCase().includes(query)
            );
        });

        container.innerHTML = filtered
            .map(
                (s) => `
        <a class="stream-card" href="/streams/${encodePathPreserveSlashes(s.path)}">
          <div class="stream-card__title">${escapeHtml(s.name)}</div>
          <div class="stream-card__path monospace">${escapeHtml(s.path)}</div>
          <div class="stream-card__meta">
            <span>Запущен: ${escapeHtml(fmtDate(s.created))}</span>
          </div>
        </a>
      `,
            )
            .join("");

        empty.classList.toggle("hidden", filtered.length !== 0);
    };

    const refresh = async () => {
        try {
            error.classList.add("hidden");
            const res = await fetch("/api/streams", { cache: "no-store" });
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data?.error || `HTTP ${res.status}`);
            }
            cachedItems = Array.isArray(data.items) ? data.items : [];
            render(cachedItems);
        } catch (e) {
            error.textContent = `Ошибка загрузки стримов: ${e.message}`;
            error.classList.remove("hidden");
            container.innerHTML = "";
            empty.classList.add("hidden");
        }
    };

    qs("#reload")?.addEventListener("click", refresh);

    let searchTimer = null;
    search?.addEventListener("input", () => {
        clearTimeout(searchTimer);
        searchTimer = setTimeout(() => render(cachedItems), 120);
    });

    await refresh();
}

function initPlayer() {
    const streamPath = document.body.dataset.streamPath;
    const video = qs("#video");
    const frame = qs("#video-shell");
    const statsEl = qs("#player-stats");

    if (!video || !frame || !statsEl) return;

    const hlsUrl = `${String(cfg.hlsBaseUrl || "").replace(/\/$/, "")}/${encodePathPreserveSlashes(streamPath)}/index.m3u8`;
    const wsUrl = `${String(cfg.captionsWsBase || "").replace(/\/$/, "")}/streams/${encodePathPreserveSlashes(streamPath)}/captions`;

    window.__STREAM_UI_STATE__ = undefined;
    const state = ensureState();

    video.setAttribute(
        "controlslist",
        "nodownload noplaybackrate noremoteplayback",
    );
    video.setAttribute("playsinline", "");
    video.setAttribute("disablepictureinpicture", "true");
    video.disablePictureInPicture = true;

    const track = video.addTextTrack("subtitles", "Русский", "ru");
    track.mode = "showing";
    state.track = track;

    renderHistory();

    const suspendUiAdjustments = () => suspendAutoAdjustments(5000);
    const uiInteractionEvents = [
        "pointerdown",
        "mousedown",
        "touchstart",
        "keydown",
        "focusin",
        "contextmenu",
    ];
    for (const eventName of uiInteractionEvents) {
        frame.addEventListener(eventName, suspendUiAdjustments, true);
        video.addEventListener(eventName, suspendUiAdjustments, true);
    }

    let hls = null;
    const ui = bindFullscreenControls(frame, video, statsEl, () => hls);

    let lastSnapAt = 0;
    const START_BUFFER_SEC = 6;
    const RESUME_BUFFER_SEC = 4;

    const getLiveEdge = () => {
        if (!video.seekable || video.seekable.length === 0) return null;
        return video.seekable.end(video.seekable.length - 1);
    };

    const snapToLive = (force = false) => {
        if (!force && !autoAdjustmentsAllowed()) return;

        const liveEdge = getLiveEdge();
        if (liveEdge == null || !Number.isFinite(liveEdge)) return;

        const drift = liveEdge - video.currentTime;
        const now = Date.now();

        if (!force) {
            if (drift < RESUME_BUFFER_SEC) return;
            if (now - lastSnapAt < 2000) return;
        }

        lastSnapAt = now;
        const target = Math.max(0, liveEdge - START_BUFFER_SEC);

        if (Math.abs(video.currentTime - target) > 0.1) {
            try {
                video.currentTime = target;
            } catch (e) {
                console.error("live snap error", e);
            }
        }
    };

    const keepStable = () => {
        if (!autoAdjustmentsAllowed()) return;
        if (
            !video.paused &&
            !video.ended &&
            Math.abs(video.playbackRate - 1) > 0.001
        ) {
            video.playbackRate = 1;
        }
    };

    const updateStats = () => {
        ui.updateStats();
    };

    const runtime = new SubtitleRuntime(video, track, updateStats);

    if (window.Hls && window.Hls.isSupported()) {
        hls = new window.Hls({
            lowLatencyMode: true,
            liveSyncDuration: 0.5,
            liveMaxLatencyDuration: 3,
            maxBufferLength: 5,
            backBufferLength: 0,
            liveBackBufferLength: 0,
            enableWorker: true,
            highBufferWatchdogPeriod: 1,
        });

        hls.loadSource(hlsUrl);
        hls.attachMedia(video);

        hls.on(window.Hls.Events.MANIFEST_PARSED, async () => {
            await waitForBuffer(video, START_BUFFER_SEC);
            snapToLive(true);
            video.play().catch(() => {});
            updateStats();
        });

        hls.on(window.Hls.Events.LEVEL_SWITCHED, () => {
            updateStats();
        });

        hls.on(window.Hls.Events.FRAG_CHANGED, () => {
            updateStats();
        });

        hls.on(window.Hls.Events.ERROR, async (_evt, data) => {
            console.error("HLS error", data);

            if (data?.details === "bufferStalledError") {
                keepStable();
                await waitForBuffer(video, START_BUFFER_SEC);
                if (video.paused && !video.ended) {
                    video.play().catch(() => {});
                }
                updateStats();
                return;
            }

            if (data?.fatal) {
                if (data.type === window.Hls.ErrorTypes.NETWORK_ERROR) {
                    try {
                        hls.startLoad();
                    } catch {}
                    updateStats();
                    return;
                }

                if (data.type === window.Hls.ErrorTypes.MEDIA_ERROR) {
                    try {
                        hls.recoverMediaError();
                    } catch {}
                    updateStats();
                }
            }
        });
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
        video.src = hlsUrl;
        video.addEventListener("loadedmetadata", async () => {
            await waitForBuffer(video, START_BUFFER_SEC);
            video.play().catch(() => {});
            updateStats();
        });
    } else {
        const status = qs("#ws-status");
        if (status) {
            status.textContent = "no hls support";
            status.dataset.state = "error";
        }
    }

    video.addEventListener("play", () => {
        keepStable();
        snapToLive(false);
        updateStats();
    });

    video.addEventListener("timeupdate", () => {
        keepStable();
        updateStats();
    });

    video.addEventListener("volumechange", () => {
        keepStable();
        updateStats();
    });

    video.addEventListener("loadeddata", updateStats);
    video.addEventListener("resize", updateStats);
    window.addEventListener("resize", updateStats);

    let ws = null;
    let retry = 0;
    let closedByUser = false;
    let reconnectTimer = null;

    const setStatus = (text, stateValue) => {
        const el = qs("#ws-status");
        if (!el) return;
        el.textContent = text;
        el.dataset.state = stateValue;
    };

    const connect = () => {
        if (closedByUser) return;

        setStatus("connecting", "connecting");
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            retry = 0;
            setStatus("connected", "ok");
        };

        ws.onmessage = (event) => {
            const raw = String(event.data ?? "");
            const parsed = safeJsonParse(raw);
            runtime.enqueue(
                parsed && typeof parsed === "object" ? parsed : raw,
            );
        };

        ws.onerror = () => {
            setStatus("error", "error");
        };

        ws.onclose = () => {
            if (closedByUser) return;
            setStatus("reconnecting", "warn");
            const delay = Math.min(5000, 500 * Math.pow(2, (retry += 1)));
            clearTimeout(reconnectTimer);
            reconnectTimer = setTimeout(connect, delay);
        };
    };

    connect();

    video.addEventListener(
        "loadedmetadata",
        () => {
            runtime.setReady();
        },
        { once: true },
    );

    const cleanup = () => {
        closedByUser = true;
        clearTimeout(reconnectTimer);
        if (ws) ws.close();
        if (hls) hls.destroy();
        for (const eventName of uiInteractionEvents) {
            frame.removeEventListener(eventName, suspendUiAdjustments, true);
            video.removeEventListener(eventName, suspendUiAdjustments, true);
        }
        ui.destroy();
    };

    window.addEventListener("beforeunload", cleanup, { once: true });

    const statsTimer = setInterval(() => {
        keepStable();
        snapToLive(false);
        updateStats();
    }, 1000);

    window.addEventListener("beforeunload", () => clearInterval(statsTimer), {
        once: true,
    });

    ui.updateStats();
}

document.addEventListener("DOMContentLoaded", () => {
    initThemeControls();

    if (document.body.dataset.page === "index") {
        loadStreams();
    }

    if (document.body.dataset.page === "stream") {
        initPlayer();
    }
});
