package model

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

type CaptionEvent struct {
	Kind                string  `json:"kind"`
	StreamName          string  `json:"stream_name"`
	CueID               string  `json:"cue_id"`
	Text                string  `json:"text"`
	Start               float64 `json:"start,omitempty"`
	End                 float64 `json:"end,omitempty"`
	EmissionTime        float64 `json:"emission_time,omitempty"`
	SourceSeq           uint64  `json:"source_seq,omitempty"`
	SourcePTS           int64   `json:"source_pts,omitempty"`
	SourceTSMS          int64   `json:"source_timestamp_ms,omitempty"`
	SourceCaptureUnixMS int64   `json:"source_capture_unix_ms,omitempty"`
	LatencyMS           int64   `json:"latency_ms,omitempty"`
}

func (c CaptionEvent) Valid() bool {
	return strings.TrimSpace(c.StreamName) != "" && strings.TrimSpace(c.Text) != ""
}

func (e CaptionEvent) WebVTT() string {
	if !e.Valid() {
		return ""
	}

	start := e.Start
	end := e.End
	if end <= 0 || end <= start {
		end = start + 4.0
	}

	var b strings.Builder
	if strings.TrimSpace(e.CueID) != "" {
		b.WriteString(strings.TrimSpace(e.CueID))
		b.WriteString("\n")
	}
	b.WriteString(formatVTTTime(start))
	b.WriteString(" --> ")
	b.WriteString(formatVTTTime(end))
	b.WriteString("\n")
	b.WriteString(e.Text)
	b.WriteString("\n\n")
	return b.String()
}

func formatVTTTime(seconds float64) string {
	if seconds < 0 {
		seconds = 0
	}
	whole := int64(seconds)
	hours := whole / 3600
	minutes := (whole % 3600) / 60
	secs := whole % 60
	millis := int64((seconds - float64(whole)) * 1000.0)
	if millis < 0 {
		millis = 0
	}
	if millis > 999 {
		millis = 999
	}
	return fmt.Sprintf("%02d:%02d:%02d.%03d", hours, minutes, secs, millis)
}

func ParseTimeStamp(raw string) (float64, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, false
	}
	parts := strings.Split(raw, ":")
	if len(parts) != 3 {
		return 0, false
	}
	h, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, false
	}
	m, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0, false
	}
	sParts := strings.SplitN(parts[2], ".", 2)
	s, err := strconv.Atoi(sParts[0])
	if err != nil {
		return 0, false
	}
	ms := 0
	if len(sParts) == 2 {
		ms, err = strconv.Atoi(padRight(sParts[1], 3))
		if err != nil {
			return 0, false
		}
	}
	return float64(h*3600+m*60+s) + float64(ms)/1000.0, true
}

func padRight(s string, n int) string {
	if len(s) >= n {
		return s[:n]
	}
	return s + strings.Repeat("0", n-len(s))
}

func secondsToDuration(seconds float64) time.Duration {
	return time.Duration(seconds * float64(time.Second))
}
