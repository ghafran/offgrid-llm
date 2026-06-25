// Package audio provides speech-to-text (ASR) and text-to-speech (TTS) capabilities
// using whisper.cpp for ASR and Piper for TTS, both running fully offline.
package audio

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

// Engine manages audio transcription and synthesis
type Engine struct {
	mu sync.RWMutex

	// Whisper settings (ASR)
	whisperPath  string
	whisperModel string
	whisperDir   string

	// Piper settings (TTS)
	piperPath  string
	piperModel string
	piperDir   string

	// General settings
	dataDir    string
	tempDir    string
	sampleRate int
}

// Config holds audio engine configuration
type Config struct {
	DataDir      string `json:"data_dir" yaml:"data_dir"`
	WhisperPath  string `json:"whisper_path" yaml:"whisper_path"`
	WhisperModel string `json:"whisper_model" yaml:"whisper_model"`
	PiperPath    string `json:"piper_path" yaml:"piper_path"`
	PiperModel   string `json:"piper_model" yaml:"piper_model"`
	SampleRate   int    `json:"sample_rate" yaml:"sample_rate"`
}

// TranscriptionRequest represents a speech-to-text request
type TranscriptionRequest struct {
	File           io.Reader `json:"-"`
	Filename       string    `json:"filename"`
	Model          string    `json:"model"`           // whisper model: tiny, base, small, medium, large
	Language       string    `json:"language"`        // language code: en, es, fr, etc.
	Prompt         string    `json:"prompt"`          // optional prompt to guide transcription
	ResponseFormat string    `json:"response_format"` // json, text, srt, vtt
	Temperature    float64   `json:"temperature"`
}

// TranscriptionResponse represents the transcription result
type TranscriptionResponse struct {
	Text     string    `json:"text"`
	Language string    `json:"language,omitempty"`
	Duration float64   `json:"duration,omitempty"`
	Segments []Segment `json:"segments,omitempty"`
}

// Segment represents a transcription segment with timing
type Segment struct {
	ID    int     `json:"id"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

// SpeechRequest represents a text-to-speech request
type SpeechRequest struct {
	Input          string  `json:"input"`           // Text to synthesize
	Model          string  `json:"model"`           // TTS model (piper model name)
	Voice          string  `json:"voice"`           // Voice name
	ResponseFormat string  `json:"response_format"` // mp3, wav, opus, flac
	Speed          float64 `json:"speed"`           // Speed multiplier (0.25 to 4.0)
}

// VoiceInfo represents available voice information
type VoiceInfo struct {
	Name        string   `json:"name"`
	Model       string   `json:"model"`
	Language    string   `json:"language"`
	Gender      string   `json:"gender"`
	SampleRate  int      `json:"sample_rate"`
	Description string   `json:"description"`
	Quality     string   `json:"quality"` // low, medium, high
	Styles      []string `json:"styles,omitempty"`
}

// NewEngine creates a new audio engine
func NewEngine(cfg Config) (*Engine, error) {
	if cfg.DataDir == "" {
		homeDir, _ := os.UserHomeDir()
		cfg.DataDir = filepath.Join(homeDir, ".offgrid-llm", "audio")
	}

	if cfg.SampleRate == 0 {
		cfg.SampleRate = 22050 // Default for Piper
	}

	e := &Engine{
		dataDir:      cfg.DataDir,
		whisperPath:  cfg.WhisperPath,
		whisperModel: cfg.WhisperModel,
		whisperDir:   filepath.Join(cfg.DataDir, "whisper"),
		piperPath:    cfg.PiperPath,
		piperModel:   cfg.PiperModel,
		piperDir:     filepath.Join(cfg.DataDir, "piper"),
		tempDir:      filepath.Join(cfg.DataDir, "temp"),
		sampleRate:   cfg.SampleRate,
	}

	// Create directories
	for _, dir := range []string{e.dataDir, e.whisperDir, e.piperDir, e.tempDir} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	// Auto-detect binaries if not specified
	if e.whisperPath == "" {
		e.whisperPath = e.findWhisper()
	}
	if e.piperPath == "" {
		e.piperPath = e.findPiper()
	}
	e.preparePiperRuntime(e.piperPath)

	return e, nil
}

// findWhisper looks for whisper binary
func (e *Engine) findWhisper() string {
	names := []string{"whisper", "whisper-cli", "whisper-cpp", "main"}
	if runtime.GOOS == "windows" {
		names = []string{"whisper.exe", "whisper-cli.exe", "whisper-cpp.exe", "main.exe"}
	}

	// Check in whisper directory
	for _, name := range names {
		path := filepath.Join(e.whisperDir, name)
		if isExecutableFile(path) {
			return path
		}
	}

	// Check in PATH
	for _, name := range names {
		if path, err := exec.LookPath(name); err == nil {
			return path
		}
	}

	return ""
}

// findPiper looks for piper binary
func (e *Engine) findPiper() string {
	names := []string{"piper", "piper-tts"}
	if runtime.GOOS == "windows" {
		names = []string{"piper.exe", "piper-tts.exe"}
	}

	// Check in piper directory and subdirectory (release extracts to piper/)
	searchDirs := []string{
		e.piperDir,
		filepath.Join(e.piperDir, "piper"),
	}

	for _, dir := range searchDirs {
		for _, name := range names {
			path := filepath.Join(dir, name)
			if isExecutableFile(path) {
				e.preparePiperRuntime(path)
				return path
			}
		}
	}

	// Check in PATH
	for _, name := range names {
		if path, err := exec.LookPath(name); err == nil {
			return path
		}
	}

	return ""
}

func (e *Engine) preparePiperRuntime(piperPath string) {
	if runtime.GOOS != "linux" {
		return
	}

	rel, err := filepath.Rel(e.piperDir, piperPath)
	if err != nil || strings.HasPrefix(rel, "..") || filepath.IsAbs(rel) {
		return
	}

	createLibSymlinks(filepath.Dir(piperPath))
}

func isRegularFile(path string) bool {
	if path == "" {
		return false
	}

	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return false
	}

	return true
}

func isExecutableFile(path string) bool {
	if !isRegularFile(path) {
		return false
	}

	if runtime.GOOS == "windows" {
		return true
	}

	info, err := os.Stat(path)
	return err == nil && info.Mode()&0111 != 0
}

// HasWhisperBinary checks if whisper binary is available
func (e *Engine) HasWhisperBinary() bool {
	return isExecutableFile(e.whisperPath)
}

// HasPiperBinary checks if piper binary is available
func (e *Engine) HasPiperBinary() bool {
	return isExecutableFile(e.piperPath)
}

// IsASRAvailable checks if speech-to-text is available
func (e *Engine) IsASRAvailable() bool {
	return e.HasWhisperBinary() && e.hasWhisperModel()
}

// IsTTSAvailable checks if text-to-speech is available
func (e *Engine) IsTTSAvailable() bool {
	return e.HasPiperBinary() && e.hasPiperModel()
}

// hasWhisperModel checks if a whisper model is available
func (e *Engine) hasWhisperModel() bool {
	if e.whisperModel != "" {
		return e.findModelPath(e.whisperModel) != ""
	}

	models, _ := e.ListWhisperModels()
	return len(models) > 0
}

// hasPiperModel checks if a piper model is available
func (e *Engine) hasPiperModel() bool {
	if e.piperModel != "" {
		return e.findPiperModelPath(e.piperModel) != ""
	}

	voices, _ := e.ListVoices()
	return len(voices) > 0
}

// Transcribe converts speech to text using Whisper
func (e *Engine) Transcribe(req TranscriptionRequest) (*TranscriptionResponse, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if !e.IsASRAvailable() {
		return nil, fmt.Errorf("ASR not available: whisper not found or no model installed")
	}

	// Get the original filename extension to determine format
	ext := ".wav"
	if req.Filename != "" {
		ext = strings.ToLower(filepath.Ext(req.Filename))
		if ext == "" {
			ext = ".wav"
		}
	}

	// Save audio to temp file with original extension
	tempFile, err := os.CreateTemp(e.tempDir, "audio-*"+ext)
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	tempPath := tempFile.Name()
	defer os.Remove(tempPath)

	if _, err := io.Copy(tempFile, req.File); err != nil {
		tempFile.Close()
		return nil, fmt.Errorf("failed to write audio: %w", err)
	}
	tempFile.Close()

	// Convert to WAV if not already WAV (whisper works best with 16kHz WAV).
	// Browser microphone recordings are usually WebM/Opus, which whisper.cpp
	// cannot decode directly.
	wavPath := tempPath
	if ext != ".wav" {
		wavPath = tempPath + ".wav"
		defer os.Remove(wavPath)

		ffmpegPath, _ := exec.LookPath("ffmpeg")
		if ffmpegPath != "" {
			cmd := exec.Command(ffmpegPath, "-y", "-i", tempPath, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wavPath)
			var stderr bytes.Buffer
			cmd.Stderr = &stderr
			if err := cmd.Run(); err != nil {
				return nil, fmt.Errorf("failed to convert %s audio to WAV with ffmpeg: %w, stderr: %s", ext, err, strings.TrimSpace(stderr.String()))
			}
		} else if isWhisperNativeAudio(ext) {
			wavPath = tempPath
		} else {
			return nil, fmt.Errorf("audio format %s requires ffmpeg for conversion; install ffmpeg or upload WAV, MP3, FLAC, or OGG audio", ext)
		}
	}

	// Build whisper command
	model := req.Model
	if model == "" {
		model = e.whisperModel
	}

	modelPath := e.findModelPath(model)
	if modelPath == "" {
		model, modelPath = e.findInstalledWhisperModel()
	}
	if modelPath == "" {
		return nil, fmt.Errorf("whisper model not found: %s", req.Model)
	}

	// Get number of CPU threads (use most available for speed)
	numCPU := runtime.NumCPU()
	if numCPU > 4 {
		numCPU = 4 // Cap at 4 for low-end machines
	}
	if numCPU < 1 {
		numCPU = 1
	}

	outputBaseFile, err := os.CreateTemp(e.tempDir, "whisper-output-*")
	if err != nil {
		return nil, fmt.Errorf("failed to create whisper output file: %w", err)
	}
	outputBase := outputBaseFile.Name()
	outputBaseFile.Close()
	os.Remove(outputBase)
	defer os.Remove(outputBase)
	defer os.Remove(outputBase + ".json")

	args := []string{
		"-m", modelPath,
		"-f", wavPath,
		"-oj", // Output JSON sidecar
		"-of", outputBase,
		"-t", fmt.Sprintf("%d", numCPU), // Use multiple threads
	}

	if req.Language != "" {
		args = append(args, "-l", req.Language)
	}

	if req.Prompt != "" {
		args = append(args, "--prompt", req.Prompt)
	}

	cmd := exec.Command(e.whisperPath, args...)

	// Set LD_LIBRARY_PATH so whisper can find its shared libraries
	whisperDir := filepath.Dir(e.whisperPath)
	ldPath := os.Getenv("LD_LIBRARY_PATH")
	if ldPath != "" {
		ldPath = whisperDir + ":" + ldPath
	} else {
		ldPath = whisperDir
	}
	cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+ldPath)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("whisper failed: %w, stderr: %s", err, stderr.String())
	}

	output := stdout.String()
	jsonOutput := []byte(output)
	if data, err := os.ReadFile(outputBase + ".json"); err == nil && len(bytes.TrimSpace(data)) > 0 {
		jsonOutput = data
	}

	if response, ok := parseWhisperJSON(jsonOutput); ok {
		return response, nil
	}

	return &TranscriptionResponse{
		Text: cleanWhisperText(output),
	}, nil
}

func isWhisperNativeAudio(ext string) bool {
	switch strings.ToLower(ext) {
	case ".flac", ".mp3", ".ogg", ".wav":
		return true
	default:
		return false
	}
}

func parseWhisperJSON(data []byte) (*TranscriptionResponse, bool) {
	var result struct {
		Result struct {
			Language string `json:"language"`
		} `json:"result"`
		Transcription []struct {
			Offsets struct {
				From int `json:"from"`
				To   int `json:"to"`
			} `json:"offsets"`
			Text string `json:"text"`
		} `json:"transcription"`
	}

	if err := json.Unmarshal(data, &result); err != nil {
		return nil, false
	}
	if result.Result.Language == "" && len(result.Transcription) == 0 {
		return nil, false
	}

	var fullText strings.Builder
	var segments []Segment
	for i, t := range result.Transcription {
		fullText.WriteString(t.Text)
		segments = append(segments, Segment{
			ID:    i,
			Start: float64(t.Offsets.From) / 1000,
			End:   float64(t.Offsets.To) / 1000,
			Text:  strings.TrimSpace(t.Text),
		})
	}

	return &TranscriptionResponse{
		Text:     strings.TrimSpace(fullText.String()),
		Language: result.Result.Language,
		Segments: segments,
	}, true
}

func cleanWhisperText(output string) string {
	output = strings.TrimSpace(output)
	if output == "" {
		return ""
	}

	lines := strings.Split(output, "\n")
	cleaned := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "[") {
			if end := strings.Index(line, "]"); end >= 0 && strings.Contains(line[:end], "-->") {
				line = strings.TrimSpace(line[end+1:])
			}
		}
		if line != "" {
			cleaned = append(cleaned, line)
		}
	}

	return strings.Join(cleaned, "\n")
}

// findModelPath finds the full path to a whisper model
func (e *Engine) findModelPath(model string) string {
	if model == "" {
		return ""
	}

	patterns := []string{
		filepath.Join(e.whisperDir, "ggml-"+model+".bin"),
		filepath.Join(e.whisperDir, model+".bin"),
		filepath.Join(e.whisperDir, model),
	}

	for _, path := range patterns {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	return ""
}

func (e *Engine) findInstalledWhisperModel() (string, string) {
	models, _ := e.ListWhisperModels()
	for _, model := range preferredWhisperModels(models) {
		if path := e.findModelPath(model); path != "" {
			return model, path
		}
	}

	return "", ""
}

// Speak converts text to speech using Piper
func (e *Engine) Speak(req SpeechRequest) (io.Reader, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if !e.IsTTSAvailable() {
		return nil, fmt.Errorf("TTS not available: piper not found or no model installed")
	}

	// Determine output format
	format := req.ResponseFormat
	if format == "" {
		format = "wav"
	}

	// Create temp output file
	tempFile, err := os.CreateTemp(e.tempDir, "speech-*."+format)
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	tempPath := tempFile.Name()
	tempFile.Close()

	// Find model - check Voice first, then Model, then default
	model := req.Voice
	if model == "" {
		model = req.Model
	}
	if model == "" {
		model = e.piperModel
	}

	modelPath := e.findPiperModelPath(model)
	if modelPath == "" {
		model, modelPath = e.findInstalledPiperVoice()
	}
	if modelPath == "" {
		return nil, fmt.Errorf("piper model not found: %s (requires both .onnx and .onnx.json files). Try running: offgrid setup piper --voice %s", req.Voice, req.Voice)
	}

	// Build piper command
	args := []string{
		"--model", modelPath,
		"--output_file", tempPath,
	}

	if req.Speed != 0 && req.Speed != 1.0 {
		args = append(args, "--length_scale", fmt.Sprintf("%.2f", 1.0/req.Speed))
	}

	cmd := exec.Command(e.piperPath, args...)
	cmd.Stdin = strings.NewReader(req.Input)

	// Set LD_LIBRARY_PATH for shared libraries (piper needs its bundled libs)
	piperDir := filepath.Dir(e.piperPath)
	cmd.Env = append(os.Environ(), fmt.Sprintf("LD_LIBRARY_PATH=%s:%s", piperDir, os.Getenv("LD_LIBRARY_PATH")))

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		os.Remove(tempPath)
		return nil, fmt.Errorf("piper failed: %w, stderr: %s", err, stderr.String())
	}

	// Read output file
	audioData, err := os.ReadFile(tempPath)
	if err != nil {
		os.Remove(tempPath)
		return nil, fmt.Errorf("failed to read audio output: %w", err)
	}
	os.Remove(tempPath)

	return bytes.NewReader(audioData), nil
}

// findPiperModelPath finds the full path to a piper model
// Returns empty string if either the .onnx or .onnx.json file is missing
func (e *Engine) findPiperModelPath(model string) string {
	if model == "" {
		return ""
	}

	patterns := []string{
		filepath.Join(e.piperDir, model+".onnx"),
		filepath.Join(e.piperDir, "voices", model+".onnx"),
		filepath.Join(e.piperDir, model, model+".onnx"),
	}

	for _, path := range patterns {
		if _, err := os.Stat(path); err == nil {
			// Also verify the JSON config exists (required by Piper)
			jsonPath := path + ".json"
			if _, err := os.Stat(jsonPath); err == nil {
				return path
			}
		}
	}

	return ""
}

func (e *Engine) findInstalledPiperVoice() (string, string) {
	voices, _ := e.ListVoices()
	for _, voice := range voices {
		if path := e.findPiperModelPath(voice.Name); path != "" {
			return voice.Name, path
		}
	}

	return "", ""
}

// ListVoices returns available TTS voices
func (e *Engine) ListVoices() ([]VoiceInfo, error) {
	var voices []VoiceInfo

	// Scan piper directory for models
	patterns := []string{
		filepath.Join(e.piperDir, "*.onnx"),
		filepath.Join(e.piperDir, "voices", "*.onnx"),
	}

	for _, pattern := range patterns {
		matches, _ := filepath.Glob(pattern)
		for _, match := range matches {
			if _, err := os.Stat(match + ".json"); err != nil {
				continue
			}

			name := strings.TrimSuffix(filepath.Base(match), ".onnx")
			voice := VoiceInfo{
				Name:  name,
				Model: match,
			}

			// Parse language from name (e.g., en_US-amy-medium)
			parts := strings.Split(name, "-")
			if len(parts) >= 1 {
				voice.Language = strings.Replace(parts[0], "_", "-", 1)
			}
			if len(parts) >= 3 {
				voice.Quality = parts[len(parts)-1]
			}

			voices = append(voices, voice)
		}
	}

	return voices, nil
}

// ListWhisperModels returns available ASR models
func (e *Engine) ListWhisperModels() ([]string, error) {
	var models []string

	pattern := filepath.Join(e.whisperDir, "*.bin")
	matches, _ := filepath.Glob(pattern)

	for _, match := range matches {
		name := filepath.Base(match)
		name = strings.TrimPrefix(name, "ggml-")
		name = strings.TrimSuffix(name, ".bin")
		models = append(models, name)
	}

	return models, nil
}

func preferredWhisperModels(models []string) []string {
	if len(models) < 2 {
		return models
	}

	preferredOrder := []string{"tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large-v3"}
	preferred := make([]string, 0, len(models))

	for _, name := range preferredOrder {
		for _, model := range models {
			if model == name {
				preferred = append(preferred, model)
				break
			}
		}
	}

	for _, model := range models {
		found := false
		for _, existing := range preferred {
			if existing == model {
				found = true
				break
			}
		}
		if !found {
			preferred = append(preferred, model)
		}
	}

	return preferred
}

// Status returns the audio engine status
func (e *Engine) Status() map[string]interface{} {
	whisperModels, _ := e.ListWhisperModels()
	voices, _ := e.ListVoices()

	return map[string]interface{}{
		"asr": map[string]interface{}{
			"available":    e.IsASRAvailable(),
			"whisper_path": e.whisperPath,
			"models":       whisperModels,
		},
		"tts": map[string]interface{}{
			"available":  e.IsTTSAvailable(),
			"piper_path": e.piperPath,
			"voices":     len(voices),
		},
		"data_dir": e.dataDir,
	}
}
