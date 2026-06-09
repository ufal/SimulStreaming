package httpserver

import (
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/websocket"

	"github.com/nukeboy76/whisper-captions/ws-captions/internal/config"
	"github.com/nukeboy76/whisper-captions/ws-captions/internal/hub"
)

func New(cfg config.Config, roomHub *hub.Hub) *http.Server {
	mux := http.NewServeMux()
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool { return true },
	}

	mux.HandleFunc("/streams/", func(w http.ResponseWriter, r *http.Request) {
		streamName, ok := streamNameFromPath(r.URL.Path)
		if !ok {
			http.Error(w, "invalid path", http.StatusBadRequest)
			return
		}

		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Println("websocket upgrade:", err)
			return
		}

		conn.SetReadLimit(cfg.WebSocketReadLimit)

		room := roomHub.GetRoom(streamName)
		client := hub.NewClient(conn)
		client.AttachRoom(room)

		if cfg.MaxRoomClients > 0 && room.Count() >= cfg.MaxRoomClients {
			_ = conn.WriteMessage(
				websocket.CloseMessage,
				websocket.FormatCloseMessage(websocket.CloseTryAgainLater, "room full"),
			)
			_ = conn.Close()
			return
		}

		room.AddClient(client)
		log.Printf("ws connected stream=%s clients=%d", streamName, room.Count())

		go writePump(client, cfg.PingInterval)
		readPump(client)

		client.Close()
		roomHub.RemoveRoomIfEmpty(streamName)
		log.Printf("ws disconnected stream=%s clients=%d", streamName, room.Count())
	})

	return &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
}

func readPump(c *hub.Client) {
	defer func() { _ = c.Conn.Close() }()

	c.Conn.SetReadLimit(1024)
	_ = c.Conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.Conn.SetPongHandler(func(string) error {
		return c.Conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	})

	for {
		if _, _, err := c.Conn.ReadMessage(); err != nil {
			return
		}
	}
}

func writePump(c *hub.Client, pingInterval time.Duration) {
	ticker := time.NewTicker(pingInterval)
	defer ticker.Stop()
	defer func() { _ = c.Conn.Close() }()

	for {
		select {
		case <-c.Done():
			return
		case msg, ok := <-c.Send:
			_ = c.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				_ = c.Conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			if err := c.Conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				return
			}
		case <-ticker.C:
			_ = c.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.Conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func streamNameFromPath(p string) (string, bool) {
	if !strings.HasPrefix(p, "/streams/") || !strings.HasSuffix(p, "/captions") {
		return "", false
	}

	trimmed := strings.TrimSuffix(strings.TrimPrefix(p, "/streams/"), "/captions")
	trimmed = strings.Trim(trimmed, "/")
	if trimmed == "" {
		return "", false
	}

	return trimmed, true
}
