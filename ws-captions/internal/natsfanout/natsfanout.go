package natsfanout

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/nats-io/nats.go"

	"github.com/nukeboy76/whisper-captions/ws-captions/internal/config"
	"github.com/nukeboy76/whisper-captions/ws-captions/internal/hub"
	"github.com/nukeboy76/whisper-captions/ws-captions/internal/model"
)

func Connect(cfg config.Config) (*nats.Conn, error) {
	nc, err := nats.Connect(
		cfg.NATSURL,
		nats.Name("ws-captions"),
		nats.MaxReconnects(-1),
		nats.ReconnectWait(2*time.Second),
	)
	if err != nil {
		return nil, fmt.Errorf("nats connect: %w", err)
	}
	return nc, nil
}

func RunSubscriber(ctx context.Context, nc *nats.Conn, roomHub *hub.Hub, cfg config.Config) error {
	_, err := nc.Subscribe(cfg.SubtitlesSubject, func(msg *nats.Msg) {
		event, ok := decodeCaptionEvent(msg)
		if !ok {
			return
		}

		payload, err := json.Marshal(event)
		if err != nil {
			return
		}

		room := roomHub.GetRoom(event.StreamName)
		room.Broadcast(payload)
	})
	if err != nil {
		return fmt.Errorf("subscribe: %w", err)
	}

	<-ctx.Done()
	return nil
}

func decodeCaptionEvent(msg *nats.Msg) (model.CaptionEvent, bool) {
	event := model.CaptionEvent{Kind: "subtitle"}

	if strings.HasPrefix(msg.Subject, "subtitles.") {
		event.StreamName = strings.TrimPrefix(msg.Subject, "subtitles.")
	}

	var raw map[string]any
	if err := json.Unmarshal(msg.Data, &raw); err != nil {
		text := strings.TrimSpace(string(msg.Data))
		if text == "" {
			return event, false
		}
		event.Text = text
		if event.StreamName == "" {
			event.StreamName = streamNameFromMsg(msg)
		}
		event.CueID = makeCueID(event.StreamName, event.Text, 0, 0)
		return event, event.Valid()
	}

	if v := stringValue(raw, "kind"); v != "" {
		event.Kind = v
	}
	if v := stringValue(raw, "stream_name"); v != "" {
		event.StreamName = v
	}
	if v := stringValue(raw, "cue_id"); v != "" {
		event.CueID = v
	}

	if payload, ok := raw["payload"].(map[string]any); ok {
		raw = payload
		if v := stringValue(raw, "stream_name"); v != "" {
			event.StreamName = v
		}
		if v := stringValue(raw, "cue_id"); v != "" {
			event.CueID = v
		}
	}

	if event.StreamName == "" {
		event.StreamName = streamNameFromMsg(msg)
	}

	event.Text = stringValue(raw, "text")
	event.Start = floatValue(raw, "start")
	event.End = floatValue(raw, "end")
	event.EmissionTime = floatValue(raw, "emission_time")
	event.SourceSeq = uint64Value(raw, "source_seq")
	event.SourcePTS = int64Value(raw, "source_pts")
	event.SourceTSMS = int64Value(raw, "source_timestamp_ms")
	event.SourceCaptureUnixMS = int64Value(raw, "source_capture_unix_ms")
	event.LatencyMS = int64Value(raw, "latency_ms")

	if event.SourceCaptureUnixMS <= 0 && msg.Header != nil {
		if v := strings.TrimSpace(msg.Header.Get("X-Source-Capture-Unix-MS")); v != "" {
			if parsed, ok := parseInt64(v); ok {
				event.SourceCaptureUnixMS = parsed
			}
		}
	}

	if event.CueID == "" {
		event.CueID = makeCueID(event.StreamName, event.Text, event.Start, event.End)
	}

	if event.Text == "" {
		return event, false
	}
	if event.StreamName == "" {
		return event, false
	}
	return event, true
}

func streamNameFromMsg(msg *nats.Msg) string {
	if msg.Header != nil {
		if v := strings.TrimSpace(msg.Header.Get("X-Stream-Name")); v != "" {
			return v
		}
	}
	if strings.HasPrefix(msg.Subject, "subtitles.") {
		return strings.TrimPrefix(msg.Subject, "subtitles.")
	}
	return ""
}

func makeCueID(streamName, text string, start, end float64) string {
	return fmt.Sprintf("%s|%.3f|%.3f|%s", streamName, start, end, strings.TrimSpace(text))
}

func stringValue(m map[string]any, key string) string {
	v, ok := m[key]
	if !ok || v == nil {
		return ""
	}
	switch x := v.(type) {
	case string:
		return x
	default:
		return fmt.Sprint(x)
	}
}

func floatValue(m map[string]any, key string) float64 {
	v, ok := m[key]
	if !ok || v == nil {
		return 0
	}
	switch x := v.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case int:
		return float64(x)
	case int64:
		return float64(x)
	case json.Number:
		n, _ := x.Float64()
		return n
	case string:
		var n float64
		_, _ = fmt.Sscan(strings.TrimSpace(x), &n)
		return n
	default:
		return 0
	}
}

func int64Value(m map[string]any, key string) int64 {
	return int64(floatValue(m, key))
}

func uint64Value(m map[string]any, key string) uint64 {
	v, ok := m[key]
	if !ok || v == nil {
		return 0
	}
	switch x := v.(type) {
	case uint64:
		return x
	case int:
		return uint64(x)
	case int64:
		return uint64(x)
	case float64:
		return uint64(x)
	case string:
		var n uint64
		_, _ = fmt.Sscan(strings.TrimSpace(x), &n)
		return n
	default:
		return 0
	}
}

func parseInt64(raw string) (int64, bool) {
	var n int64
	_, err := fmt.Sscan(strings.TrimSpace(raw), &n)
	if err != nil {
		return 0, false
	}
	return n, true
}
