package inference

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
)

const binaryInstallFormatVersion = "2"

// BinaryManager handles the lifecycle of the llama-server binary
type BinaryManager struct {
	version   string
	binDir    string
	hasNVIDIA bool
	hasAMD    bool
}

// NewBinaryManager creates a new binary manager
func NewBinaryManager(binDir string) *BinaryManager {
	bm := &BinaryManager{
		version: getLlamaVersion(),
		binDir:  binDir,
	}
	bm.detectGPU()
	return bm
}

func getLlamaVersion() string {
	if version := strings.TrimSpace(os.Getenv("OFFGRID_LLAMA_CPP_VERSION")); version != "" {
		return version
	}
	return "b9787"
}

// detectGPU checks for available GPU acceleration
func (bm *BinaryManager) detectGPU() {
	// Check for NVIDIA
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		cmd := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
		if output, err := cmd.Output(); err == nil && len(output) > 0 {
			bm.hasNVIDIA = true
		}
	}

	// Check for AMD ROCm
	if _, err := exec.LookPath("rocm-smi"); err == nil {
		bm.hasAMD = true
	}
}

// HasGPU returns true if GPU acceleration is available
func (bm *BinaryManager) HasGPU() bool {
	return bm.hasNVIDIA || bm.hasAMD
}

// GPUType returns the detected GPU type
func (bm *BinaryManager) GPUType() string {
	if bm.hasNVIDIA {
		return "nvidia"
	}
	if bm.hasAMD {
		return "amd"
	}
	return "cpu"
}

// GetLlamaServer returns the path to the llama-server binary
// It checks (in order):
// 1. OFFGRID_LLAMA_SERVER_PATH environment variable
// 2. Local bin directory
// 3. System PATH
// 4. Downloads it if not found
func (bm *BinaryManager) GetLlamaServer() (string, error) {
	// 1. Check env
	if path := os.Getenv("OFFGRID_LLAMA_SERVER_PATH"); path != "" {
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}

	// 2. Check local bin
	binaryName := "llama-server"
	if runtime.GOOS == "windows" {
		binaryName += ".exe"
	}
	localPath := filepath.Join(bm.binDir, binaryName)
	if bm.isManagedBinaryCurrent(localPath) {
		return localPath, nil
	}

	// 3. Check PATH
	if path, err := exec.LookPath("llama-server"); err == nil {
		return path, nil
	}

	// 4. Download
	if err := bm.downloadBinary(localPath); err != nil {
		return "", fmt.Errorf("failed to download llama-server: %w", err)
	}

	return localPath, nil
}

func (bm *BinaryManager) isManagedBinaryCurrent(localPath string) bool {
	info, err := os.Stat(localPath)
	if err != nil || info.IsDir() {
		return false
	}

	versionBytes, err := os.ReadFile(bm.versionStampPath())
	if err != nil {
		return false
	}

	return strings.TrimSpace(string(versionBytes)) == bm.version+"\t"+binaryInstallFormatVersion
}

func (bm *BinaryManager) versionStampPath() string {
	return filepath.Join(bm.binDir, "llama-server.version")
}

func (bm *BinaryManager) writeVersionStamp() error {
	return os.WriteFile(bm.versionStampPath(), []byte(bm.version+"\t"+binaryInstallFormatVersion+"\n"), 0644)
}

func (bm *BinaryManager) downloadBinary(destPath string) error {
	// Ensure bin dir exists
	if err := os.MkdirAll(bm.binDir, 0755); err != nil {
		return err
	}

	url, err := bm.getDownloadURL()
	if err != nil {
		return bm.buildFromSource(destPath, err)
	}

	fmt.Println()
	fmt.Printf("  ⇣ Downloading llama-server (one-time setup)...\n")
	fmt.Printf("    Source: %s\n", url)
	if bm.hasNVIDIA {
		fmt.Printf("    GPU:    NVIDIA (CUDA acceleration enabled)\n")
	} else if bm.hasAMD {
		fmt.Printf("    GPU:    AMD (ROCm acceleration enabled)\n")
	} else {
		fmt.Printf("    GPU:    None (CPU mode)\n")
	}
	fmt.Println()

	archiveExt := ".zip"
	if strings.HasSuffix(url, ".tar.gz") {
		archiveExt = ".tar.gz"
	}

	// Download archive to temp file
	tmpFile, err := os.CreateTemp("", "llama-server-*"+archiveExt)
	if err != nil {
		return err
	}
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("download failed: HTTP %s", resp.Status)
		if resp.StatusCode == http.StatusNotFound {
			return bm.buildFromSource(destPath, err)
		}
		return err
	}

	// Get content length for progress
	contentLength := resp.ContentLength

	// Download with progress
	var downloaded int64
	buf := make([]byte, 32*1024)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := tmpFile.Write(buf[:n]); writeErr != nil {
				return writeErr
			}
			downloaded += int64(n)
			if contentLength > 0 {
				percent := float64(downloaded) / float64(contentLength) * 100
				fmt.Printf("\r    Progress: %.1f%% (%d MB / %d MB)", percent, downloaded/(1024*1024), contentLength/(1024*1024))
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}
	fmt.Println()
	fmt.Printf("    ✓ Download complete\n")

	// Extract archive
	fmt.Printf("    Extracting binary...\n")
	if strings.HasSuffix(url, ".tar.gz") {
		if err := extractBinaryTarGz(tmpFile.Name(), destPath); err != nil {
			return err
		}
	} else {
		if err := bm.extractZip(tmpFile.Name(), destPath); err != nil {
			return err
		}
	}
	if err := bm.writeVersionStamp(); err != nil {
		return fmt.Errorf("failed to write llama-server version stamp: %w", err)
	}
	fmt.Printf("    ✓ Installed to %s\n", destPath)
	fmt.Println()

	return nil
}

func (bm *BinaryManager) getDownloadURL() (string, error) {
	return bm.getDownloadURLFor(runtime.GOOS, runtime.GOARCH)
}

func (bm *BinaryManager) getDownloadURLFor(goos, goarch string) (string, error) {
	baseURL := "https://github.com/ggml-org/llama.cpp/releases/download"

	// Map OS/Arch/GPU to asset name
	// Based on llama.cpp release conventions as of late 2024
	var assetName string

	switch goos {
	case "linux":
		if goarch == "amd64" {
			assetName = fmt.Sprintf("llama-%s-bin-ubuntu-x64.tar.gz", bm.version)
		} else if goarch == "arm64" {
			assetName = fmt.Sprintf("llama-%s-bin-ubuntu-arm64.tar.gz", bm.version)
		}
	case "darwin":
		if goarch == "arm64" {
			// Apple Silicon with Metal support
			assetName = fmt.Sprintf("llama-%s-bin-macos-arm64.tar.gz", bm.version)
		} else if goarch == "amd64" {
			assetName = fmt.Sprintf("llama-%s-bin-macos-x64.tar.gz", bm.version)
		}
	case "windows":
		if goarch == "amd64" {
			if bm.hasNVIDIA {
				// CUDA build for Windows with NVIDIA
				assetName = fmt.Sprintf("llama-%s-bin-win-cuda-12.4-x64.zip", bm.version)
			} else {
				assetName = fmt.Sprintf("llama-%s-bin-win-cpu-x64.zip", bm.version)
			}
		} else if goarch == "arm64" {
			assetName = fmt.Sprintf("llama-%s-bin-win-cpu-arm64.zip", bm.version)
		}
	}

	if assetName == "" {
		return "", fmt.Errorf("no prebuilt llama-server asset for platform: %s/%s", goos, goarch)
	}

	return fmt.Sprintf("%s/%s/%s", baseURL, bm.version, assetName), nil
}

func (bm *BinaryManager) buildFromSource(destPath string, prebuiltErr error) error {
	if !supportsLlamaSourceBuild(runtime.GOOS, runtime.GOARCH) {
		return prebuiltErr
	}
	if err := ensureLlamaBuildTools(); err != nil {
		return fmt.Errorf("%w; source build fallback unavailable: %w", prebuiltErr, err)
	}

	fmt.Println()
	fmt.Printf("  ⚙ Building llama-server from source (one-time setup)...\n")
	fmt.Printf("    Reason: %v\n", prebuiltErr)
	fmt.Printf("    Version: %s\n", bm.version)
	fmt.Println()

	tmpDir, err := os.MkdirTemp("", "llama-cpp-build-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	archivePath := filepath.Join(tmpDir, "llama.cpp.tar.gz")
	sourceURL := fmt.Sprintf("https://github.com/ggml-org/llama.cpp/archive/refs/tags/%s.tar.gz", bm.version)
	fmt.Printf("    Source: %s\n", sourceURL)
	if err := downloadFile(sourceURL, archivePath); err != nil {
		return fmt.Errorf("failed to download llama.cpp source: %w", err)
	}

	sourceDir, err := extractTarGz(archivePath, tmpDir)
	if err != nil {
		return fmt.Errorf("failed to extract llama.cpp source: %w", err)
	}

	buildDir := filepath.Join(tmpDir, "build")
	cmakeArgs := []string{
		"-S", sourceDir,
		"-B", buildDir,
		"-DGGML_STATIC=ON",
		"-DBUILD_SHARED_LIBS=OFF",
		"-DLLAMA_BUILD_SERVER=ON",
		"-DLLAMA_CURL=OFF",
	}
	if _, err := exec.LookPath("ninja"); err == nil {
		cmakeArgs = append([]string{"-G", "Ninja"}, cmakeArgs...)
	}

	if err := runBuildCommand("cmake", cmakeArgs...); err != nil {
		return fmt.Errorf("failed to configure llama.cpp: %w", err)
	}

	buildArgs := []string{"--build", buildDir, "--config", "Release", "--target", "llama-server", "-j", fmt.Sprintf("%d", runtime.NumCPU())}
	if err := runBuildCommand("cmake", buildArgs...); err != nil {
		return fmt.Errorf("failed to build llama-server: %w", err)
	}

	builtPath, err := findBuiltLlamaServer(buildDir)
	if err != nil {
		return err
	}

	if err := copyFile(builtPath, destPath); err != nil {
		return fmt.Errorf("failed to install llama-server: %w", err)
	}
	if err := os.Chmod(destPath, 0755); err != nil {
		return fmt.Errorf("failed to make llama-server executable: %w", err)
	}
	if err := bm.writeVersionStamp(); err != nil {
		return fmt.Errorf("failed to write llama-server version stamp: %w", err)
	}

	fmt.Printf("    ✓ Built and installed to %s\n", destPath)
	fmt.Println()

	return nil
}

func supportsLlamaSourceBuild(goos, goarch string) bool {
	switch goos {
	case "linux", "darwin":
		return goarch == "amd64" || goarch == "arm64"
	default:
		return false
	}
}

func ensureLlamaBuildTools() error {
	required := []string{"cmake"}
	for _, tool := range required {
		if _, err := exec.LookPath(tool); err != nil {
			return fmt.Errorf("%s is required to build llama.cpp", tool)
		}
	}

	if _, err := exec.LookPath("c++"); err != nil {
		if _, gppErr := exec.LookPath("g++"); gppErr != nil {
			if _, clangErr := exec.LookPath("clang++"); clangErr != nil {
				return fmt.Errorf("a C++ compiler is required to build llama.cpp")
			}
		}
	}

	if _, err := exec.LookPath("ninja"); err == nil {
		return nil
	}
	if _, err := exec.LookPath("make"); err != nil {
		return fmt.Errorf("make or ninja is required to build llama.cpp")
	}

	return nil
}

func downloadFile(url, destPath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %s", resp.Status)
	}

	out, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func extractTarGz(archivePath, destDir string) (string, error) {
	f, err := os.Open(archivePath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return "", err
	}
	defer gz.Close()

	tr := tar.NewReader(gz)
	var rootDir string

	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		if header.Typeflag == tar.TypeXGlobalHeader || header.Typeflag == tar.TypeXHeader {
			continue
		}

		cleanName := path.Clean(header.Name)
		if cleanName == "." || cleanName == ".." || strings.HasPrefix(cleanName, "../") || path.IsAbs(cleanName) {
			return "", fmt.Errorf("unsafe path in archive: %s", header.Name)
		}

		switch header.Typeflag {
		case tar.TypeDir:
			parts := strings.Split(cleanName, "/")
			if len(parts) > 0 && parts[0] != "." && rootDir == "" {
				rootDir = filepath.Join(destDir, parts[0])
			}
			targetPath := filepath.Join(append([]string{destDir}, parts...)...)
			if err := os.MkdirAll(targetPath, os.FileMode(header.Mode)); err != nil {
				return "", err
			}
		case tar.TypeReg:
			parts := strings.Split(cleanName, "/")
			if len(parts) > 0 && parts[0] != "." && rootDir == "" {
				rootDir = filepath.Join(destDir, parts[0])
			}
			targetPath := filepath.Join(append([]string{destDir}, parts...)...)
			if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
				return "", err
			}
			out, err := os.OpenFile(targetPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, os.FileMode(header.Mode))
			if err != nil {
				return "", err
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				return "", err
			}
			if err := out.Close(); err != nil {
				return "", err
			}
		}
	}

	if rootDir == "" {
		return "", fmt.Errorf("source archive did not contain a root directory")
	}
	return rootDir, nil
}

func extractBinaryTarGz(archivePath, destPath string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	binaryName := "llama-server"
	if runtime.GOOS == "windows" {
		binaryName += ".exe"
	}
	destDir := filepath.Dir(destPath)

	tr := tar.NewReader(gz)
	foundBinary := false
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		cleanName := path.Clean(header.Name)
		if cleanName == "." || cleanName == ".." || strings.HasPrefix(cleanName, "../") || path.IsAbs(cleanName) {
			return fmt.Errorf("unsafe path in archive: %s", header.Name)
		}
		baseName := path.Base(cleanName)
		if baseName == "." || baseName == "/" {
			continue
		}

		targetPath := filepath.Join(destDir, baseName)
		if baseName == binaryName || baseName == "server" || baseName == "server.exe" {
			targetPath = destPath
			foundBinary = true
		}

		switch header.Typeflag {
		case tar.TypeReg:
			mode := os.FileMode(header.Mode)
			if mode == 0 {
				mode = 0644
			}

			out, err := os.OpenFile(targetPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, mode)
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				return err
			}
			if err := out.Close(); err != nil {
				return err
			}
		case tar.TypeSymlink:
			linkName := path.Clean(header.Linkname)
			if linkName == "." || linkName == ".." || strings.HasPrefix(linkName, "../") || path.IsAbs(linkName) || strings.Contains(linkName, "/") {
				return fmt.Errorf("unsafe symlink in archive: %s -> %s", header.Name, header.Linkname)
			}
			if err := os.RemoveAll(targetPath); err != nil {
				return err
			}
			if err := os.Symlink(linkName, targetPath); err != nil {
				return err
			}
		default:
			continue
		}
	}

	if !foundBinary {
		return fmt.Errorf("binary %s not found in tar.gz", binaryName)
	}

	return os.Chmod(destPath, 0755)
}

func runBuildCommand(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Env = os.Environ()
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s %s failed: %w\n%s", name, strings.Join(args, " "), err, trimBuildOutput(string(output)))
	}
	return nil
}

func trimBuildOutput(output string) string {
	const maxOutput = 4000
	if len(output) <= maxOutput {
		return output
	}
	return output[len(output)-maxOutput:]
}

func findBuiltLlamaServer(buildDir string) (string, error) {
	names := []string{"llama-server", "server"}
	if runtime.GOOS == "windows" {
		names = []string{"llama-server.exe", "server.exe"}
	}

	searchDirs := []string{
		filepath.Join(buildDir, "bin"),
		filepath.Join(buildDir, "examples", "server"),
		buildDir,
	}
	for _, dir := range searchDirs {
		for _, name := range names {
			path := filepath.Join(dir, name)
			if info, err := os.Stat(path); err == nil && !info.IsDir() {
				return path, nil
			}
		}
	}

	return "", fmt.Errorf("llama-server binary not found after build")
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0755)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	return err
}

// GetVersion returns the pinned llama.cpp version
func (bm *BinaryManager) GetVersion() string {
	return bm.version
}

// IsInstalled checks if llama-server is already installed
func (bm *BinaryManager) IsInstalled() bool {
	binaryName := "llama-server"
	if runtime.GOOS == "windows" {
		binaryName += ".exe"
	}
	localPath := filepath.Join(bm.binDir, binaryName)
	_, err := os.Stat(localPath)
	return err == nil
}

// GetInstalledPath returns the path to the installed binary, or empty if not installed
func (bm *BinaryManager) GetInstalledPath() string {
	binaryName := "llama-server"
	if runtime.GOOS == "windows" {
		binaryName += ".exe"
	}
	localPath := filepath.Join(bm.binDir, binaryName)
	if _, err := os.Stat(localPath); err == nil {
		return localPath
	}
	if path, err := exec.LookPath("llama-server"); err == nil {
		return path
	}
	return ""
}

func (bm *BinaryManager) extractZip(zipPath, destPath string) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer r.Close()

	binaryName := "llama-server"
	if runtime.GOOS == "windows" {
		binaryName += ".exe"
	}
	destDir := filepath.Dir(destPath)
	foundBinary := false

	for _, f := range r.File {
		if f.FileInfo().IsDir() {
			continue
		}

		cleanName := path.Clean(strings.ReplaceAll(f.Name, "\\", "/"))
		if cleanName == "." || cleanName == ".." || strings.HasPrefix(cleanName, "../") || path.IsAbs(cleanName) {
			return fmt.Errorf("unsafe path in zip: %s", f.Name)
		}
		baseName := path.Base(cleanName)
		if baseName == "." || baseName == "/" {
			continue
		}

		targetPath := filepath.Join(destDir, baseName)
		if baseName == binaryName || baseName == "server" || baseName == "server.exe" {
			targetPath = destPath
			foundBinary = true
		}

		rc, err := f.Open()
		if err != nil {
			return err
		}

		mode := f.FileInfo().Mode()
		if mode == 0 {
			mode = 0644
		}
		outFile, err := os.OpenFile(targetPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, mode)
		if err != nil {
			rc.Close()
			return err
		}
		if _, err = io.Copy(outFile, rc); err != nil {
			outFile.Close()
			rc.Close()
			return err
		}
		if err := outFile.Close(); err != nil {
			rc.Close()
			return err
		}
		if err := rc.Close(); err != nil {
			return err
		}
	}

	if !foundBinary {
		return fmt.Errorf("binary %s not found in zip", binaryName)
	}

	return os.Chmod(destPath, 0755)
}
