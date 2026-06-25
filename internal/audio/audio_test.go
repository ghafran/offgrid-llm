package audio

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestAvailabilityUsesInstalledAudioAssets(t *testing.T) {
	engine := newTestEngine(t)
	engine.whisperPath = writeTestExecutable(t, filepath.Join(engine.dataDir, "bin", "whisper"))
	engine.piperPath = writeTestExecutable(t, filepath.Join(engine.dataDir, "bin", "piper"))

	writeTestFile(t, filepath.Join(engine.whisperDir, "ggml-large-v3.bin"))
	writeTestFile(t, filepath.Join(engine.piperDir, "en_US-libritts-high.onnx"))
	writeTestFile(t, filepath.Join(engine.piperDir, "en_US-libritts-high.onnx.json"))

	if !engine.IsASRAvailable() {
		t.Fatal("ASR should be available when any whisper model is installed")
	}
	if !engine.IsTTSAvailable() {
		t.Fatal("TTS should be available when any complete piper voice is installed")
	}

	model, modelPath := engine.findInstalledWhisperModel()
	if model != "large-v3" || modelPath == "" {
		t.Fatalf("installed whisper fallback = (%q, %q), want large-v3 and path", model, modelPath)
	}

	voice, voicePath := engine.findInstalledPiperVoice()
	if voice != "en_US-libritts-high" || voicePath == "" {
		t.Fatalf("installed piper fallback = (%q, %q), want en_US-libritts-high and path", voice, voicePath)
	}
}

func TestAvailabilityHonorsConfiguredMissingAssets(t *testing.T) {
	engine := newTestEngine(t)
	engine.whisperPath = writeTestExecutable(t, filepath.Join(engine.dataDir, "bin", "whisper"))
	engine.piperPath = writeTestExecutable(t, filepath.Join(engine.dataDir, "bin", "piper"))
	engine.whisperModel = "base.en"
	engine.piperModel = "en_US-amy-medium"

	writeTestFile(t, filepath.Join(engine.whisperDir, "ggml-large-v3.bin"))
	writeTestFile(t, filepath.Join(engine.piperDir, "en_US-libritts-high.onnx"))
	writeTestFile(t, filepath.Join(engine.piperDir, "en_US-libritts-high.onnx.json"))

	if engine.IsASRAvailable() {
		t.Fatal("ASR should not be available when a configured whisper model is missing")
	}
	if engine.IsTTSAvailable() {
		t.Fatal("TTS should not be available when a configured piper voice is missing")
	}
}

func TestAvailabilityRejectsDirectoryBinaryPath(t *testing.T) {
	engine := newTestEngine(t)
	engine.piperPath = filepath.Join(engine.piperDir, "piper")

	if err := os.MkdirAll(engine.piperPath, 0755); err != nil {
		t.Fatalf("failed to create piper directory: %v", err)
	}
	writeTestFile(t, filepath.Join(engine.piperDir, "en_US-valid-medium.onnx"))
	writeTestFile(t, filepath.Join(engine.piperDir, "en_US-valid-medium.onnx.json"))

	if engine.HasPiperBinary() {
		t.Fatal("directory path should not count as a piper binary")
	}
	if engine.IsTTSAvailable() {
		t.Fatal("TTS should not be available when piper path is a directory")
	}
}

func TestFindPiperSkipsExtractedDirectory(t *testing.T) {
	engine := newTestEngine(t)
	nestedPiper := writeTestExecutable(t, filepath.Join(engine.piperDir, "piper", "piper"))
	writeTestFile(t, filepath.Join(engine.piperDir, "piper", "libespeak-ng.so.1.52.0.1"))
	writeTestFile(t, filepath.Join(engine.piperDir, "piper", "libpiper_phonemize.so.1.2.0"))

	if got := engine.findPiper(); got != nestedPiper {
		t.Fatalf("findPiper() = %q, want %q", got, nestedPiper)
	}
	if runtime.GOOS == "linux" {
		for _, name := range []string{"libespeak-ng.so.1", "libpiper_phonemize.so.1"} {
			if _, err := os.Lstat(filepath.Join(engine.piperDir, "piper", name)); err != nil {
				t.Fatalf("expected piper runtime symlink %s: %v", name, err)
			}
		}
	}
}

func TestListVoicesRequiresJsonConfig(t *testing.T) {
	engine := newTestEngine(t)

	writeTestFile(t, filepath.Join(engine.piperDir, "en_US-valid-medium.onnx"))
	writeTestFile(t, filepath.Join(engine.piperDir, "en_US-valid-medium.onnx.json"))
	writeTestFile(t, filepath.Join(engine.piperDir, "en_US-missing-json-medium.onnx"))

	voices, err := engine.ListVoices()
	if err != nil {
		t.Fatalf("ListVoices returned error: %v", err)
	}
	if len(voices) != 1 {
		t.Fatalf("len(voices) = %d, want 1", len(voices))
	}
	if voices[0].Name != "en_US-valid-medium" {
		t.Fatalf("voice = %q, want en_US-valid-medium", voices[0].Name)
	}
}

func TestPreferredWhisperModels(t *testing.T) {
	models := preferredWhisperModels([]string{"large-v3", "base.en", "tiny.en"})
	want := []string{"tiny.en", "base.en", "large-v3"}

	if len(models) != len(want) {
		t.Fatalf("len(models) = %d, want %d", len(models), len(want))
	}
	for i := range want {
		if models[i] != want[i] {
			t.Fatalf("models[%d] = %q, want %q", i, models[i], want[i])
		}
	}
}

func TestTranscribeRejectsWebMWithoutFFmpeg(t *testing.T) {
	engine := newTestEngine(t)
	engine.whisperPath = writeTestExecutable(t, filepath.Join(engine.dataDir, "bin", "whisper"))
	writeTestFile(t, filepath.Join(engine.whisperDir, "ggml-tiny.en.bin"))
	t.Setenv("PATH", t.TempDir())

	_, err := engine.Transcribe(TranscriptionRequest{
		File:     strings.NewReader("webm data"),
		Filename: "voice.webm",
		Model:    "tiny.en",
	})
	if err == nil {
		t.Fatal("Transcribe should reject WebM when ffmpeg is not available")
	}
	if !strings.Contains(err.Error(), "requires ffmpeg") {
		t.Fatalf("error = %q, want ffmpeg guidance", err.Error())
	}
}

func TestParseWhisperJSON(t *testing.T) {
	response, ok := parseWhisperJSON([]byte(`{
		"result": {"language": "en"},
		"transcription": [
			{"offsets": {"from": 0, "to": 1250}, "text": " Hello"},
			{"offsets": {"from": 1250, "to": 2500}, "text": " world."}
		]
	}`))
	if !ok {
		t.Fatal("parseWhisperJSON should parse whisper.cpp JSON output")
	}
	if response.Text != "Hello world." {
		t.Fatalf("Text = %q, want %q", response.Text, "Hello world.")
	}
	if response.Language != "en" {
		t.Fatalf("Language = %q, want en", response.Language)
	}
	if len(response.Segments) != 2 {
		t.Fatalf("len(Segments) = %d, want 2", len(response.Segments))
	}
	if response.Segments[1].Start != 1.25 || response.Segments[1].End != 2.5 || response.Segments[1].Text != "world." {
		t.Fatalf("Segments[1] = %+v, want start/end/text from JSON", response.Segments[1])
	}
}

func TestCleanWhisperTextRemovesTimestamps(t *testing.T) {
	got := cleanWhisperText("[00:00:00.000 --> 00:00:01.000]   Hello\n[00:00:01.000 --> 00:00:02.000] world")
	want := "Hello\nworld"
	if got != want {
		t.Fatalf("cleanWhisperText() = %q, want %q", got, want)
	}
}

func newTestEngine(t *testing.T) *Engine {
	t.Helper()

	engine, err := NewEngine(Config{DataDir: t.TempDir()})
	if err != nil {
		t.Fatalf("NewEngine returned error: %v", err)
	}
	return engine
}

func writeTestFile(t *testing.T, path string) {
	t.Helper()

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("failed to create test dir: %v", err)
	}
	if err := os.WriteFile(path, []byte("test"), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}
}

func writeTestExecutable(t *testing.T, path string) string {
	t.Helper()

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("failed to create test dir: %v", err)
	}
	if err := os.WriteFile(path, []byte("test"), 0755); err != nil {
		t.Fatalf("failed to write test executable: %v", err)
	}
	if runtime.GOOS != "windows" {
		if err := os.Chmod(path, 0755); err != nil {
			t.Fatalf("failed to chmod test executable: %v", err)
		}
	}
	return path
}
