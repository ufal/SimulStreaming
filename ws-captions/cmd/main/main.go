package main

import (
	"context"
	"log"
	"os/signal"
	"syscall"

	"github.com/nukeboy76/whisper-captions/ws-captions/internal/config"
	"github.com/nukeboy76/whisper-captions/ws-captions/internal/httpserver"
	"github.com/nukeboy76/whisper-captions/ws-captions/internal/hub"
	"github.com/nukeboy76/whisper-captions/ws-captions/internal/natsfanout"
)

func main() {
	cfg := config.Load()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	roomHub := hub.New()

	nc, err := natsfanout.Connect(cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer nc.Drain()

	go func() {
		if err := natsfanout.RunSubscriber(ctx, nc, roomHub, cfg); err != nil {
			log.Println("nats subscriber:", err)
			stop()
		}
	}()

	srv := httpserver.New(cfg, roomHub)
	go func() {
		<-ctx.Done()
		if err := srv.Shutdown(context.Background()); err != nil {
			log.Println("http shutdown:", err)
		}
	}()

	log.Println("ws-captions started on", cfg.ListenAddr)
	if err := srv.ListenAndServe(); err != nil {
		log.Println("http server:", err)
	}
	log.Println("ws-captions stopped")
}
