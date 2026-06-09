package server

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"sync"

	"github.com/nukeboy76/whisper-captions/audio-segmenter/internal/stream"
	"github.com/nukeboy76/whisper-captions/audio-segmenter/internal/worker"
)

func Server() *http.ServeMux {
	log.Println("audio-segmenter server started")

	mux := http.NewServeMux()

	var (
		mu      sync.Mutex
		streams = make(map[string]context.CancelFunc)
	)

	mux.HandleFunc("/stream-started", func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body error", http.StatusBadRequest)
			log.Println("/stream-started read body error:", err)
			return
		}

		log.Printf("/stream-started raw body: %s", string(body))

		var strm stream.Stream
		if err := json.Unmarshal(body, &strm); err != nil {
			http.Error(w, "invalid json", http.StatusBadRequest)
			log.Println("/stream-started unmarshal error:", err)
			return
		}

		log.Printf(
			"/stream-started parsed rtsp=%q path=%q source_type=%q source_id=%q",
			strm.RTSPURL,
			strm.Path,
			strm.SourceType,
			strm.SourceID,
		)

		if strm.RTSPURL == "" {
			http.Error(w, "rtsp_url is required", http.StatusBadRequest)
			return
		}

		ctx, cancel := context.WithCancel(context.Background())

		mu.Lock()
		if oldCancel, ok := streams[strm.RTSPURL]; ok {
			oldCancel()
		}
		streams[strm.RTSPURL] = cancel
		mu.Unlock()

		go worker.Listen(ctx, strm.RTSPURL, strm.Path)

		w.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc("/stream-stopped", func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body error", http.StatusBadRequest)
			log.Println("/stream-stopped read body error:", err)
			return
		}

		log.Printf("/stream-stopped raw body: %s", string(body))

		var strm stream.Stream
		if err := json.Unmarshal(body, &strm); err != nil {
			http.Error(w, "invalid json", http.StatusBadRequest)
			log.Println("/stream-stopped unmarshal error:", err)
			return
		}

		log.Printf(
			"/stream-stopped parsed rtsp=%q path=%q source_type=%q source_id=%q",
			strm.RTSPURL,
			strm.Path,
			strm.SourceType,
			strm.SourceID,
		)

		mu.Lock()
		cancel, ok := streams[strm.RTSPURL]
		if ok {
			cancel()
			delete(streams, strm.RTSPURL)
		}
		mu.Unlock()

		if !ok {
			http.Error(w, "stream not found", http.StatusNotFound)
			return
		}

		w.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	return mux
}
