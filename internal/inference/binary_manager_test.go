package inference

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestBinaryManager_GetLlamaServer_Local(t *testing.T) {
	// Create temp bin dir
	tmpDir, err := os.MkdirTemp("", "offgrid_bin_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	bm := NewBinaryManager(tmpDir)

	// Create dummy binary
	binaryName := "llama-server"
	if runtime.GOOS == "windows" {
		binaryName += ".exe"
	}
	binaryPath := filepath.Join(tmpDir, binaryName)

	if err := os.WriteFile(binaryPath, []byte("dummy"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := bm.writeVersionStamp(); err != nil {
		t.Fatal(err)
	}

	// Test finding it
	foundPath, err := bm.GetLlamaServer()
	if err != nil {
		t.Fatalf("Failed to find local binary: %v", err)
	}

	if foundPath != binaryPath {
		t.Errorf("Expected path %s, got %s", binaryPath, foundPath)
	}
}

func TestBinaryManager_GetDownloadURL(t *testing.T) {
	bm := NewBinaryManager("/tmp")
	url, err := bm.getDownloadURL()
	if err != nil {
		// Might fail on unsupported platforms, which is fine
		t.Logf("getDownloadURL returned error (might be expected): %v", err)
	} else {
		t.Logf("Download URL: %s", url)
		if url == "" {
			t.Error("URL should not be empty")
		}
	}
}

func TestBinaryManager_GetDownloadURLForLinuxArm64(t *testing.T) {
	bm := NewBinaryManager("/tmp")
	url, err := bm.getDownloadURLFor("linux", "arm64")
	if err != nil {
		t.Fatalf("getDownloadURLFor(linux, arm64) returned error: %v", err)
	}
	if want := "llama-" + bm.GetVersion() + "-bin-ubuntu-arm64.tar.gz"; !strings.Contains(url, want) {
		t.Fatalf("url = %q, want asset %q", url, want)
	}
}

func TestBinaryManager_GetDownloadURLForUnsupportedPlatformFallsBackToSource(t *testing.T) {
	bm := NewBinaryManager("/tmp")
	url, err := bm.getDownloadURLFor("linux", "riscv64")

	if err == nil {
		t.Fatalf("getDownloadURLFor(linux, riscv64) = %q, want source-build fallback error", url)
	}
	if !strings.Contains(err.Error(), "no prebuilt llama-server asset") {
		t.Fatalf("error = %q, want no prebuilt asset message", err)
	}
}

func TestBinaryManager_GetDownloadURLForLinuxAmd64(t *testing.T) {
	bm := NewBinaryManager("/tmp")
	url, err := bm.getDownloadURLFor("linux", "amd64")
	if err != nil {
		t.Fatalf("getDownloadURLFor(linux, amd64) returned error: %v", err)
	}
	if want := "llama-" + bm.GetVersion() + "-bin-ubuntu-x64.tar.gz"; !strings.Contains(url, want) {
		t.Fatalf("url = %q, want asset %q", url, want)
	}
}

func TestExtractTarGzIgnoresPaxGlobalHeader(t *testing.T) {
	tmpDir := t.TempDir()
	archivePath := filepath.Join(tmpDir, "source.tar.gz")

	file, err := os.Create(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	gz := gzip.NewWriter(file)
	tw := tar.NewWriter(gz)

	if err := tw.WriteHeader(&tar.Header{
		Name:     "pax_global_header",
		Typeflag: tar.TypeXGlobalHeader,
	}); err != nil {
		t.Fatal(err)
	}
	if err := tw.WriteHeader(&tar.Header{
		Name:     "llama.cpp-b4320/",
		Typeflag: tar.TypeDir,
		Mode:     0755,
	}); err != nil {
		t.Fatal(err)
	}
	if err := tw.WriteHeader(&tar.Header{
		Name:     "llama.cpp-b4320/CMakeLists.txt",
		Typeflag: tar.TypeReg,
		Mode:     0644,
		Size:     int64(len("cmake")),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write([]byte("cmake")); err != nil {
		t.Fatal(err)
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}

	extractDir := filepath.Join(tmpDir, "extract")
	root, err := extractTarGz(archivePath, extractDir)
	if err != nil {
		t.Fatalf("extractTarGz returned error: %v", err)
	}
	if root != filepath.Join(extractDir, "llama.cpp-b4320") {
		t.Fatalf("root = %q, want llama.cpp source root", root)
	}
	if _, err := os.Stat(filepath.Join(root, "CMakeLists.txt")); err != nil {
		t.Fatalf("expected extracted source file: %v", err)
	}
}

func TestExtractBinaryTarGz(t *testing.T) {
	tmpDir := t.TempDir()
	archivePath := filepath.Join(tmpDir, "llama-bin.tar.gz")

	file, err := os.Create(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	gz := gzip.NewWriter(file)
	tw := tar.NewWriter(gz)

	content := []byte("binary")
	if err := tw.WriteHeader(&tar.Header{
		Name:     "llama-bin/bin/llama-server",
		Typeflag: tar.TypeReg,
		Mode:     0755,
		Size:     int64(len(content)),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write(content); err != nil {
		t.Fatal(err)
	}
	libContent := []byte("library")
	if err := tw.WriteHeader(&tar.Header{
		Name:     "llama-bin/bin/libllama-server-impl.so",
		Typeflag: tar.TypeReg,
		Mode:     0644,
		Size:     int64(len(libContent)),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write(libContent); err != nil {
		t.Fatal(err)
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}

	destPath := filepath.Join(tmpDir, "llama-server")
	if err := extractBinaryTarGz(archivePath, destPath); err != nil {
		t.Fatalf("extractBinaryTarGz returned error: %v", err)
	}
	got, err := os.ReadFile(destPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(content) {
		t.Fatalf("extracted content = %q, want %q", got, content)
	}
	if _, err := os.Stat(filepath.Join(tmpDir, "libllama-server-impl.so")); err != nil {
		t.Fatalf("expected bundled shared library: %v", err)
	}
}

func TestExtractZipInstallsSidecarLibraries(t *testing.T) {
	tmpDir := t.TempDir()
	archivePath := filepath.Join(tmpDir, "llama-bin.zip")

	file, err := os.Create(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(file)

	binaryWriter, err := zw.Create("llama-bin/bin/llama-server")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := binaryWriter.Write([]byte("binary")); err != nil {
		t.Fatal(err)
	}
	libWriter, err := zw.Create("llama-bin/bin/llama.dll")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := libWriter.Write([]byte("library")); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}

	bm := NewBinaryManager(tmpDir)
	destPath := filepath.Join(tmpDir, "llama-server")
	if err := bm.extractZip(archivePath, destPath); err != nil {
		t.Fatalf("extractZip returned error: %v", err)
	}
	if _, err := os.Stat(destPath); err != nil {
		t.Fatalf("expected extracted llama-server: %v", err)
	}
	if _, err := os.Stat(filepath.Join(tmpDir, "llama.dll")); err != nil {
		t.Fatalf("expected bundled shared library: %v", err)
	}
}
