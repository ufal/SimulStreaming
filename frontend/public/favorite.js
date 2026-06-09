(() => {
    const STORAGE_KEY = "stream-ui-favorites";
    const TAB_KEY = "stream-ui-favorites-tab";

    const state = {
        tab: "all",
        favorites: new Set(),
        renderTimer: null,
        observer: null,
        ready: false,
        emptyText: "Стримов пока нет",
        toolbar: null,
    };

    const qs = (selector, root = document) => root.querySelector(selector);

    function isIndexPage() {
        return document.body?.dataset?.page === "index";
    }

    function readJson(key, fallback) {
        try {
            const raw = localStorage.getItem(key);
            if (!raw) return fallback;
            return JSON.parse(raw);
        } catch {
            return fallback;
        }
    }

    function saveJson(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch {}
    }

    function loadFavorites() {
        const items = readJson(STORAGE_KEY, []);
        if (!Array.isArray(items)) return new Set();
        return new Set(items.filter((v) => typeof v === "string" && v.trim()));
    }

    function saveFavorites() {
        saveJson(STORAGE_KEY, [...state.favorites]);
    }

    function loadTab() {
        const value = readJson(TAB_KEY, "all");
        return value === "favorites" ? "favorites" : "all";
    }

    function saveTab() {
        saveJson(TAB_KEY, state.tab);
    }

    function injectStyles() {
        if (qs("#favorite-ui-styles")) return;

        const style = document.createElement("style");
        style.id = "favorite-ui-styles";
        style.textContent = `
            .favorite-toolbar {
                display: flex;
                align-items: stretch;
                justify-content: space-between;
                gap: 12px;
                flex-wrap: wrap;
            }

            .favorite-tabs {
                display: flex;
                gap: 0;
                flex-wrap: nowrap;
                width: 100%;
            }

            .favorite-tab {
                flex: 1 1 0;
                min-width: 0;
                justify-content: center;
                border: 0 !important;
                outline: 0 !important;
                appearance: none;
                -webkit-appearance: none;
                -moz-appearance: none;
                background: linear-gradient(180deg, var(--panel), var(--panel-2)) !important;
                box-shadow: inset 0 0 0 1px var(--line-strong);
                border-radius: 0;
                position: relative;
                z-index: 0;
            }

            .favorite-tab + .favorite-tab {
                margin-left: -1px;
            }

            .favorite-tab:hover {
                box-shadow: inset 0 0 0 1px var(--line-strong);
            }

            .favorite-tab[data-active="true"] {
                box-shadow: inset 0 0 0 2px var(--favorite-active-border);
                z-index: 1;
                font-weight: 700;
                color: var(--favorite-active-border);
            }

            .favorite-tab:focus,
            .favorite-tab:focus-visible {
                outline: 0 !important;
            }

            .favorite-card-shell {
                display: flex;
                align-items: stretch;
                gap: 10px;
            }

            .favorite-card-shell > .stream-card {
                flex: 1 1 auto;
            }

            .favorite-toggle {
                min-width: 48px;
                width: 48px;
                padding: 0;
                display: grid;
                place-items: center;
                font-size: 18px;
                line-height: 1;
            }

            .favorite-toggle[data-active="true"] {
                font-weight: 700;
                color: var(--favorite-active-border);
            }
            .favorite-toggle[data-active="false"] {
                color: var(--text);
            }

            .favorite-card-shell.favorite-hidden {
                display: none !important;
            }
        `;
        document.head.appendChild(style);
    }

    function ensureToolbar() {
        if (state.toolbar) return state.toolbar;

        const listPanel = qs(".list-panel");
        if (!listPanel) return null;

        const toolbar = document.createElement("section");
        toolbar.className = "favorite-toolbar";
        toolbar.id = "favorite-toolbar";
        toolbar.innerHTML = `
            <div class="favorite-tabs" role="tablist" aria-label="Фильтр стримов">
                <button type="button" class="button favorite-tab" data-tab="all" aria-pressed="true">Все</button>
                <button type="button" class="button favorite-tab" data-tab="favorites" aria-pressed="false">Избранное</button>
            </div>
        `;

        listPanel.parentElement.insertBefore(toolbar, listPanel);
        state.toolbar = toolbar;

        toolbar.querySelectorAll(".favorite-tab").forEach((button) => {
            button.addEventListener("click", () => {
                const next =
                    button.dataset.tab === "favorites" ? "favorites" : "all";
                setTab(next);
            });
        });

        return toolbar;
    }

    function setTab(tab) {
        state.tab = tab === "favorites" ? "favorites" : "all";
        saveTab();
        applyView();
    }

    function getStreamsContainer() {
        return qs("#streams");
    }

    function getErrorEl() {
        return qs("#error");
    }

    function getEmptyEl() {
        return qs("#empty");
    }

    function getRenderedCards() {
        const container = getStreamsContainer();
        if (!container) return [];
        return [...container.children].filter(
            (node) => node instanceof HTMLElement,
        );
    }

    function isErrorVisible() {
        const errorEl = getErrorEl();
        return !!errorEl && !errorEl.classList.contains("hidden");
    }

    function extractPathFromCard(card) {
        if (!(card instanceof HTMLElement)) return "";
        if (card.dataset.streamPath) return card.dataset.streamPath;

        const link = card.matches("a.stream-card")
            ? card
            : qs("a.stream-card", card);
        const href = link?.getAttribute("href") || "";
        const match = href.match(/^\/streams\/(.+)$/);
        if (!match) return "";
        try {
            return decodeURIComponent(match[1]);
        } catch {
            return match[1];
        }
    }

    function getShellForNode(node) {
        if (!(node instanceof HTMLElement)) return null;
        if (node.classList.contains("favorite-card-shell")) return node;
        return null;
    }

    function wrapCards() {
        const container = getStreamsContainer();
        if (!container) return;

        const children = [...container.children];
        for (const child of children) {
            if (!(child instanceof HTMLElement)) continue;
            if (child.classList.contains("favorite-card-shell")) continue;
            if (!child.matches("a.stream-card")) continue;

            const path = extractPathFromCard(child);
            const shell = document.createElement("div");
            shell.className = "favorite-card-shell";
            shell.dataset.streamPath = path;

            const favButton = document.createElement("button");
            favButton.type = "button";
            favButton.className = "button favorite-toggle";
            favButton.dataset.streamPath = path;
            favButton.setAttribute("aria-label", "Добавить в избранное");

            favButton.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();
                toggleFavorite(path);
            });

            child.dataset.streamPath = path;
            child.style.display = "";
            child.style.flex = "1 1 auto";

            container.replaceChild(shell, child);
            shell.appendChild(child);
            shell.appendChild(favButton);
        }
    }

    function normalizeFavoritesAgainstCurrentStreams() {
        const renderedPaths = new Set();

        for (const node of getRenderedCards()) {
            const shell = getShellForNode(node);
            const path = shell?.dataset.streamPath || extractPathFromCard(node);
            if (path) renderedPaths.add(path);
        }

        if (!renderedPaths.size) return;

        const next = new Set();
        for (const favorite of state.favorites) {
            if (renderedPaths.has(favorite)) {
                next.add(favorite);
            }
        }

        if (next.size !== state.favorites.size) {
            state.favorites = next;
            saveFavorites();
        }
    }

    function updateToolbarState() {
        const toolbar = ensureToolbar();
        if (!toolbar) return;

        const count = state.favorites.size;

        toolbar.querySelectorAll(".favorite-tab").forEach((button) => {
            const tab =
                button.dataset.tab === "favorites" ? "favorites" : "all";
            const active = tab === state.tab;

            button.dataset.active = String(active);
            button.setAttribute("aria-pressed", String(active));

            if (tab === "favorites") {
                button.textContent =
                    count > 0 ? `Избранное (${count})` : "Избранное";
            } else {
                button.textContent = "Все";
            }
        });
    }

    function applyView() {
        wrapCards();

        const cards = getRenderedCards();
        let visibleFavorites = 0;
        let visibleAll = 0;

        for (const node of cards) {
            const shell = getShellForNode(node);
            if (!shell) continue;

            const path = shell.dataset.streamPath || extractPathFromCard(node);
            const isFavorite = state.favorites.has(path);
            const shouldHide =
                state.tab === "favorites" && !isFavorite && !isErrorVisible();

            shell.classList.toggle("favorite-hidden", shouldHide);
            shell.style.display = shouldHide ? "none" : "flex";

            const button = qs(".favorite-toggle", shell);
            if (button) {
                button.dataset.active = String(isFavorite);
                button.textContent = isFavorite ? "★" : "☆";
                button.setAttribute(
                    "aria-label",
                    isFavorite
                        ? "Убрать из избранного"
                        : "Добавить в избранное",
                );
            }

            if (!shouldHide) {
                visibleAll += 1;
            }

            if (isFavorite && !shouldHide) {
                visibleFavorites += 1;
            }
        }

        const emptyEl = getEmptyEl();
        if (emptyEl) {
            if (state.tab === "favorites") {
                if (visibleFavorites === 0 && !isErrorVisible()) {
                    emptyEl.textContent = "Избранных стримов пока нет";
                    emptyEl.classList.remove("hidden");
                } else {
                    emptyEl.textContent = state.emptyText;
                    emptyEl.classList.add("hidden");
                }
            } else {
                if (visibleAll === 0 && !isErrorVisible()) {
                    emptyEl.textContent = state.emptyText;
                    emptyEl.classList.remove("hidden");
                } else {
                    emptyEl.textContent = state.emptyText;
                    emptyEl.classList.add("hidden");
                }
            }
        }

        updateToolbarState();
    }

    function toggleFavorite(path) {
        if (!path) return;

        if (state.favorites.has(path)) {
            state.favorites.delete(path);
        } else {
            state.favorites.add(path);
        }

        saveFavorites();
        applyView();
    }

    function scheduleApply() {
        clearTimeout(state.renderTimer);
        state.renderTimer = setTimeout(() => {
            if (!isIndexPage()) return;

            const emptyEl = getEmptyEl();
            if (emptyEl && !state.ready) {
                state.emptyText = emptyEl.textContent || state.emptyText;
                state.ready = true;
            }

            ensureToolbar();
            normalizeFavoritesAgainstCurrentStreams();
            applyView();
        }, 30);
    }

    function observeStreams() {
        const container = getStreamsContainer();
        if (!container) return;

        if (state.observer) {
            state.observer.disconnect();
        }

        state.observer = new MutationObserver(() => {
            scheduleApply();
        });

        state.observer.observe(container, {
            childList: true,
            subtree: false,
        });
    }

    function syncFromStorage() {
        state.tab = loadTab();
        state.favorites = loadFavorites();
        updateToolbarState();
        applyView();
    }

    function init() {
        if (!isIndexPage()) return;

        injectStyles();
        state.tab = loadTab();
        state.favorites = loadFavorites();
        ensureToolbar();
        observeStreams();
        syncFromStorage();
        scheduleApply();

        window.addEventListener("storage", (event) => {
            if (event.key === STORAGE_KEY || event.key === TAB_KEY) {
                state.tab = loadTab();
                state.favorites = loadFavorites();
                syncFromStorage();
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init, { once: true });
    } else {
        init();
    }
})();
