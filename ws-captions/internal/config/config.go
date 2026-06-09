package config

import (
	"os"
	"strconv"
	"time"
)

type Config struct {
	ListenAddr         string
	NATSURL            string
	SubtitlesSubject   string
	MaxRoomClients     int
	WebSocketReadLimit int64
	PingInterval       time.Duration
}

func Load() Config {
	return Config{
		ListenAddr:         env("LISTEN_ADDR", ":8002"),
		NATSURL:            env("NATS_URL", "nats://localhost:4222"),
		SubtitlesSubject:   env("SUBTITLES_SUBJECT", "subtitles.*"),
		MaxRoomClients:     envInt("MAX_ROOM_CLIENTS", 0),
		WebSocketReadLimit: int64(envInt("WS_READ_LIMIT", 4096)),
		PingInterval:       envDuration("WS_PING_INTERVAL", 30*time.Second),
	}
}

func env(key, fallback string) string {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	return v
}

func envInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}

func envDuration(key string, fallback time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return fallback
	}
	return d
}
