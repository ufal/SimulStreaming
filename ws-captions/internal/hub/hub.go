package hub

import (
	"sync"

	"github.com/gorilla/websocket"
)

type Client struct {
	Conn      *websocket.Conn
	Send      chan []byte
	room      *Room
	done      chan struct{}
	closeOnce sync.Once
}

func NewClient(conn *websocket.Conn) *Client {
	return &Client{
		Conn: conn,
		Send: make(chan []byte, 64),
		done: make(chan struct{}),
	}
}

func (c *Client) AttachRoom(r *Room)    { c.room = r }
func (c *Client) Done() <-chan struct{} { return c.done }

func (c *Client) Close() {
	c.closeOnce.Do(func() {
		close(c.done)
		if c.room != nil {
			c.room.RemoveClient(c)
		}
		_ = c.Conn.Close()
	})
}

type Room struct {
	Name    string
	mu      sync.RWMutex
	clients map[*Client]struct{}
}

func newRoom(name string) *Room {
	return &Room{
		Name:    name,
		clients: make(map[*Client]struct{}),
	}
}

func (r *Room) AddClient(c *Client) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.clients[c] = struct{}{}
}

func (r *Room) RemoveClient(c *Client) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.clients, c)
}

func (r *Room) Snapshot() []*Client {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Client, 0, len(r.clients))
	for c := range r.clients {
		out = append(out, c)
	}
	return out
}

func (r *Room) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.clients)
}

func (r *Room) Broadcast(msg []byte) {
	for _, c := range r.Snapshot() {
		select {
		case <-c.Done():
			continue
		default:
		}
		select {
		case c.Send <- msg:
		default:
			c.Close()
		}
	}
}

type Hub struct {
	mu    sync.RWMutex
	rooms map[string]*Room
}

func New() *Hub {
	return &Hub{rooms: make(map[string]*Room)}
}

func (h *Hub) GetRoom(name string) *Room {
	h.mu.Lock()
	defer h.mu.Unlock()
	if room, ok := h.rooms[name]; ok {
		return room
	}
	room := newRoom(name)
	h.rooms[name] = room
	return room
}

func (h *Hub) RemoveRoomIfEmpty(name string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	room, ok := h.rooms[name]
	if !ok {
		return
	}
	if room.Count() == 0 {
		delete(h.rooms, name)
	}
}
