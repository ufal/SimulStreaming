package stream

type Stream struct {
	RTSPURL    string `json:"RTSPURL"`
	Path       string `json:"path"`
	SourceType string `json:"source_type"`
	SourceID   string `json:"source_id"`
}
