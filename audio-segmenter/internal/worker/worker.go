package worker

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/url"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/bluenviron/gortsplib/v5"
	"github.com/bluenviron/gortsplib/v5/pkg/base"
	"github.com/bluenviron/gortsplib/v5/pkg/format"
	"github.com/nats-io/nats.go"
	"github.com/pion/rtp"
	"gopkg.in/hraban/opus.v2"
)

const (
	defaultNATSURL       = "nats://localhost:4222"
	audioSubjectPrefix   = "opus.segments."
	streamActivePrefix   = "streams.active."
	streamInactivePrefix = "streams.inactive."
	audioStreamName      = "OPUS_SEGMENTS"
	streamEventsName     = "STREAM_EVENTS"

	inputSampleRate  = 48000
	outputSampleRate = 16000
	audioChannels    = 1
	chunkDurationMs  = 200
	audioCodec       = "pcm_s16le"
)

type audioChunk struct {
	StreamName    string
	Seq           uint16
	Timestamp     uint32
	CaptureUnixMS int64
	PTS           int64
	Payload       []byte
}

func Listen(ctx context.Context, streamURL string, streamNameOverride string) {
	natsURL := natsURL()
	streamName := streamNameOverride
	if streamName == "" {
		streamName = streamNameFromURL(streamURL)
	}

	log.Printf("listen start stream=%q url=%q", streamName, streamURL)

	nc, err := nats.Connect(
		natsURL,
		nats.Name("audio-segmenter"),
		nats.Timeout(5*time.Second),
	)
	if err != nil {
		log.Println("nats connect:", err)
		return
	}
	defer nc.Close()

	js, err := nc.JetStream()
	if err != nil {
		log.Println("jetstream init:", err)
		return
	}

	if err := ensureStreams(js); err != nil {
		log.Println("ensure streams:", err)
		return
	}

	if err := publishLifecycle(js, streamActivePrefix+safeKeyPart(streamName), streamName, streamURL); err != nil {
		log.Println("stream active publish error:", err)
	}
	defer func() {
		if err := publishLifecycle(js, streamInactivePrefix+safeKeyPart(streamName), streamName, streamURL); err != nil {
			log.Println("stream inactive publish error:", err)
		}
	}()

	u, err := base.ParseURL(streamURL)
	if err != nil {
		log.Println("rtsp url parse:", err)
		return
	}

	c := &gortsplib.Client{
		Scheme: u.Scheme,
		Host:   u.Host,
	}

	proto := gortsplib.ProtocolTCP
	c.Protocol = &proto

	if err := c.Start(); err != nil {
		log.Println("rtsp start:", err)
		return
	}
	defer c.Close()

	log.Printf("rtsp client started stream=%q host=%s", streamName, u.Host)

	desc, _, err := c.Describe(u)
	if err != nil {
		log.Println("rtsp describe:", err)
		return
	}

	var opusFormat *format.Opus
	media := desc.FindFormat(&opusFormat)
	if media == nil {
		log.Println("opus track not found")
		return
	}

	log.Printf("rtsp describe ok stream=%q media=%s clock=%d", streamName, media.Type, opusFormat.ClockRate())

	if _, err := c.Setup(desc.BaseURL, media, 0, 0); err != nil {
		log.Println("rtsp setup:", err)
		return
	}

	rtpDecoder, err := opusFormat.CreateDecoder()
	if err != nil {
		log.Println("rtp opus decoder init:", err)
		return
	}

	opusDecoder, err := opus.NewDecoder(inputSampleRate, audioChannels)
	if err != nil {
		log.Println("opus decoder init:", err)
		return
	}

	chunks := make(chan audioChunk, 256)
	go publishChunks(ctx, js, chunks)

	var seq uint16
	var ptsSamples int64
	var packetsSeen uint64
	startedAt := time.Now()

	c.OnPacketRTP(media, opusFormat, func(pkt *rtp.Packet) {
		if pkt == nil || len(pkt.Payload) == 0 {
			return
		}

		if atomic.AddUint64(&packetsSeen, 1) == 1 {
			log.Printf("first rtp packet stream=%q payload_bytes=%d", streamName, len(pkt.Payload))
		}

		_, ok := c.PacketPTS(media, pkt)
		if !ok {
			log.Printf("packet pts unavailable stream=%q seq=%d", streamName, pkt.SequenceNumber)
			return
		}

		opkt, err := rtpDecoder.Decode(pkt)
		if err != nil {
			log.Printf("rtp depacketize error stream=%q seq=%d err=%v", streamName, pkt.SequenceNumber, err)
			return
		}

		if len(opkt) == 0 {
			return
		}

		pcm48k, err := decodeOpusPacket(opusDecoder, opkt)
		if err != nil {
			log.Printf("opus decode error stream=%q seq=%d err=%v", streamName, pkt.SequenceNumber, err)
			return
		}

		if len(pcm48k) == 0 {
			return
		}

		pcm16k := resampleLinearFloat32(pcm48k, inputSampleRate, outputSampleRate)
		if len(pcm16k) == 0 {
			return
		}

		raw := float32ToPCM16LE(pcm16k)
		if len(raw) == 0 {
			return
		}

		for len(raw) >= chunkBytes() {
			payload := append([]byte(nil), raw[:chunkBytes()]...)
			raw = raw[chunkBytes():]

			captureUnixMS := time.Now().UnixMilli()

			chunks <- audioChunk{
				StreamName:    streamName,
				Seq:           seq,
				Timestamp:     uint32(time.Since(startedAt).Milliseconds()),
				CaptureUnixMS: captureUnixMS,
				PTS:           ptsSamples,
				Payload:       payload,
			}

			log.Printf(
				"chunk queued stream=%q seq=%d pts=%d ts=%d capture_unix_ms=%d bytes=%d",
				streamName,
				seq,
				ptsSamples,
				time.Since(startedAt).Milliseconds(),
				captureUnixMS,
				len(payload),
			)

			seq++
			ptsSamples += int64(len(payload) / 2)
		}

		if len(raw) > 0 {
			if len(raw)%2 != 0 {
				raw = raw[:len(raw)-1]
			}
			if len(raw) > 0 {
				captureUnixMS := time.Now().UnixMilli()

				chunks <- audioChunk{
					StreamName:    streamName,
					Seq:           seq,
					Timestamp:     uint32(time.Since(startedAt).Milliseconds()),
					CaptureUnixMS: captureUnixMS,
					PTS:           ptsSamples,
					Payload:       append([]byte(nil), raw...),
				}

				seq++
				ptsSamples += int64(len(raw) / 2)
			}
		}
	})

	if _, err := c.Play(nil); err != nil {
		log.Println("rtsp play:", err)
		close(chunks)
		return
	}

	log.Printf("rtsp play ok stream=%q", streamName)

	done := make(chan error, 1)
	go func() {
		done <- c.Wait()
	}()

	select {
	case <-ctx.Done():
		log.Printf("context canceled stream=%q", streamName)
		close(chunks)
		<-done
	case err := <-done:
		if err != nil && !errors.Is(err, io.EOF) {
			log.Println("rtsp wait:", err)
		}
		close(chunks)
	}
}

func decodeOpusPacket(dec *opus.Decoder, data []byte) ([]float32, error) {
	pcm := make([]int16, 5760)

	n, err := dec.Decode(data, pcm)
	if err != nil {
		return nil, err
	}
	if n <= 0 {
		return nil, nil
	}

	pcm = pcm[:n]

	out := make([]float32, len(pcm))
	for i, sample := range pcm {
		out[i] = float32(sample) / 32768.0
	}

	return out, nil
}

func publishChunks(ctx context.Context, js nats.JetStreamContext, chunks <-chan audioChunk) {
	for {
		select {
		case <-ctx.Done():
			return
		case ch, ok := <-chunks:
			if !ok {
				return
			}

			subject := audioSubjectPrefix + safeKeyPart(ch.StreamName)
			msgBytes := encodeAudioChunk(ch)

			msg := nats.NewMsg(subject)
			msg.Data = msgBytes
			msg.Header.Set("X-Stream-Name", ch.StreamName)
			msg.Header.Set("X-Stream-Key", safeKeyPart(ch.StreamName))
			msg.Header.Set("X-Audio-Codec", audioCodec)
			msg.Header.Set("X-Sample-Rate", fmt.Sprintf("%d", outputSampleRate))
			msg.Header.Set("X-Channels", fmt.Sprintf("%d", audioChannels))
			msg.Header.Set("X-Seq", fmt.Sprintf("%d", ch.Seq))
			msg.Header.Set("X-Timestamp", fmt.Sprintf("%d", ch.Timestamp))
			msg.Header.Set("X-Capture-Unix-MS", fmt.Sprintf("%d", ch.CaptureUnixMS))
			msg.Header.Set("X-PTS", fmt.Sprintf("%d", ch.PTS))
			msg.Header.Set("X-Payload-Bytes", fmt.Sprintf("%d", len(ch.Payload)))

			if _, err := js.PublishMsg(msg); err != nil {
				log.Printf(
					"nats publish error: subject=%s stream_name=%s seq=%d pts=%d ts=%d capture_unix_ms=%d payload_bytes=%d err=%v",
					subject,
					ch.StreamName,
					ch.Seq,
					ch.PTS,
					ch.Timestamp,
					ch.CaptureUnixMS,
					len(ch.Payload),
					err,
				)
				continue
			}
		}
	}
}

func chunkBytes() int {
	return (outputSampleRate * chunkDurationMs / 1000) * 2
}

func resampleLinearFloat32(in []float32, inRate, outRate int) []float32 {
	if len(in) == 0 {
		return nil
	}
	if inRate == outRate {
		out := make([]float32, len(in))
		copy(out, in)
		return out
	}

	ratio := float64(inRate) / float64(outRate)
	outLen := int(math.Round(float64(len(in)) / ratio))
	if outLen < 1 {
		outLen = 1
	}

	out := make([]float32, outLen)
	for i := 0; i < outLen; i++ {
		pos := float64(i) * ratio
		idx := int(pos)
		frac := float32(pos - float64(idx))

		if idx >= len(in)-1 {
			out[i] = in[len(in)-1]
			continue
		}

		a := in[idx]
		b := in[idx+1]
		out[i] = a*(1-frac) + b*frac
	}
	return out
}

func float32ToPCM16LE(samples []float32) []byte {
	if len(samples) == 0 {
		return nil
	}

	out := make([]byte, len(samples)*2)
	for i, s := range samples {
		if s > 1 {
			s = 1
		}
		if s < -1 {
			s = -1
		}
		v := int16(s * 32767)
		binary.LittleEndian.PutUint16(out[i*2:], uint16(v))
	}
	return out
}

func ensureStreams(js nats.JetStreamContext) error {
	if err := ensureStream(js, audioStreamName, []string{"opus.segments.*"}, 24*time.Hour); err != nil {
		return err
	}
	if err := ensureStream(js, streamEventsName, []string{"streams.active.*", "streams.inactive.*"}, 24*time.Hour); err != nil {
		return err
	}
	return nil
}

func ensureStream(js nats.JetStreamContext, name string, subjects []string, maxAge time.Duration) error {
	cfg := &nats.StreamConfig{
		Name:      name,
		Subjects:  subjects,
		Storage:   nats.FileStorage,
		Retention: nats.LimitsPolicy,
		Discard:   nats.DiscardOld,
		MaxAge:    maxAge,
	}

	info, err := js.StreamInfo(name)
	if err == nil {
		if !sameSubjects(info.Config.Subjects, subjects) {
			_, err = js.UpdateStream(cfg)
			if err != nil {
				return fmt.Errorf("update stream %s: %w", name, err)
			}
		}
		return nil
	}

	if !errors.Is(err, nats.ErrStreamNotFound) {
		return err
	}

	_, err = js.AddStream(cfg)
	if err != nil {
		return fmt.Errorf("add stream %s: %w", name, err)
	}

	return nil
}

func publishLifecycle(js nats.JetStreamContext, subject, streamName, streamURL string) error {
	msg := nats.NewMsg(subject)
	msg.Data = []byte(streamURL)
	msg.Header.Set("X-Stream-Name", streamName)
	msg.Header.Set("X-Stream-Key", safeKeyPart(streamName))
	msg.Header.Set("X-Stream-URL", streamURL)
	msg.Header.Set("X-Event-Time", time.Now().UTC().Format(time.RFC3339Nano))
	_, err := js.PublishMsg(msg)
	return err
}

func encodeAudioChunk(ch audioChunk) []byte {
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.BigEndian, ch.Seq)
	_ = binary.Write(&buf, binary.BigEndian, ch.Timestamp)
	_ = binary.Write(&buf, binary.BigEndian, ch.PTS)
	_ = binary.Write(&buf, binary.BigEndian, uint32(len(ch.Payload)))
	_, _ = buf.Write(ch.Payload)
	return buf.Bytes()
}

func natsURL() string {
	raw := strings.TrimSpace(os.Getenv("NATS_URL"))
	if raw == "" {
		return defaultNATSURL
	}
	return raw
}

func streamNameFromURL(streamURL string) string {
	u, err := url.Parse(streamURL)
	if err != nil {
		return "stream"
	}

	name := strings.TrimPrefix(u.Path, "/")
	name = strings.Trim(name, "/")
	if name == "" {
		return "stream"
	}
	return name
}

func safeKeyPart(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var b strings.Builder
	b.Grow(len(s))

	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '_' || r == '-' || r == '.':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}

	out := strings.Trim(b.String(), "._-")
	if out == "" {
		return "stream"
	}
	return out
}

func sameSubjects(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
