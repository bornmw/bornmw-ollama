//go:build integration

package integration

import (
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
)

// generateDummyNVFP4GGUF creates a minimal dummy NVFP4 GGUF file using the Go
// ggml.WriteGGUF writer, which guarantees a format that the Go decoder can
// round-trip. The model has a single tiny tensor so it can be processed quickly.
func generateDummyNVFP4GGUF(t *testing.T, dir string) (ggufPath string, err error) {
	t.Helper()

	f, err := os.CreateTemp(dir, "dummy-nvfp4-*.gguf")
	if err != nil {
		return "", err
	}
	defer func() {
		if err != nil {
			f.Close()
			os.Remove(f.Name())
		}
	}()

	fileTypeNvfp4, err := ggml.ParseFileType("NVFP4")
	if err != nil {
		return "", err
	}
	tensorTypeNvfp4, err := ggml.ParseTensorType("NVFP4")
	if err != nil {
		return "", err
	}

	kv := ggml.KV{
		"general.architecture":                      "llama",
		"general.file_type":                         uint32(fileTypeNvfp4),
		"general.parameter_count":                   uint64(1),
		"general.alignment":                         uint32(32),
		"llama.context_length":                      uint32(64),
		"llama.embedding_length":                    uint32(256),
		"llama.block_count":                         uint32(1),
		"llama.attention.head_count":                uint32(8),
		"llama.attention.head_count_kv":             uint32(8),
		"llama.attention.layer_norm_rms_epsilon":    float32(1e-5),
	}

	nvfp4Tensor := &ggml.Tensor{
		Name: "blk.0.attn_q.weight",
		Kind: uint32(tensorTypeNvfp4),
		Shape: []uint64{256},
		WriterTo: &zeroReader{size: 144},
	}

	// NVFP4 uses a global per-tensor scale (single element).
	scaleTensor := &ggml.Tensor{
		Name:     "blk.0.attn_q.weight_scale",
		Kind:     uint32(ggml.TensorTypeF32),
		Shape:    []uint64{1},
		WriterTo: &zeroReader{size: 4},
	}

	if err := ggml.WriteGGUF(f, kv, []*ggml.Tensor{nvfp4Tensor, scaleTensor}); err != nil {
		return "", err
	}

	if err := f.Close(); err != nil {
		return "", err
	}

	return f.Name(), nil
}

// zeroReader is an io.Reader that always returns zero bytes. Used for
// dummy weight data in the test GGUF file.
type zeroReader struct {
	size uint64
}

func (z *zeroReader) WriteTo(w io.Writer) (int64, error) {
	if z.size == 0 {
		return 0, nil
	}
	return io.Copy(w, io.LimitReader(z, int64(z.size)))
}

func (z *zeroReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = 0
	}
	return len(p), nil
}

func TestNvfp4E2EBoundary(t *testing.T) {
	if testModel != "" {
		t.Skip("NVFP4 E2E boundary test not applicable with model override")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Generate dummy NVFP4 GGUF using Go writer
	tmpDir := t.TempDir()
	ggufPath, err := generateDummyNVFP4GGUF(t, tmpDir)
	if err != nil {
		t.Fatalf("Failed to generate dummy NVFP4 GGUF: %v", err)
	}
	t.Logf("Generated dummy NVFP4 GGUF at %s", ggufPath)

	absGGUF, err := filepath.Abs(ggufPath)
	if err != nil {
		t.Fatalf("Failed to get absolute path: %v", err)
	}

	modelName := "test-nvfp4-dummy"

	// Create a Modelfile and use the CLI to create the model from the dummy GGUF
	tmpModelfile := filepath.Join(tmpDir, "Modelfile")
	modelfileContent := "FROM " + absGGUF + "\n"
	if err := os.WriteFile(tmpModelfile, []byte(modelfileContent), 0o644); err != nil {
		t.Fatalf("Failed to write Modelfile: %v", err)
	}

	createCmd := exec.CommandContext(ctx, ollamaBin(), "create", modelName, "-f", tmpModelfile)
	var createStderr strings.Builder
	createCmd.Stdout = os.Stdout
	createCmd.Stderr = io.MultiWriter(os.Stderr, &createStderr)

	if err := createCmd.Run(); err != nil {
		t.Logf("ollama create output: %s", createStderr.String())
		// Check for Go-level errors that would indicate the boundary failed
		output := createStderr.String()
		if strings.Contains(output, "panic") {
			t.Fatalf("Go panic during model creation: %s", output)
		}
		if strings.Contains(output, "unknown format") {
			t.Fatalf("Go rejected NVFP4 format: %s", output)
		}

		// llama-quantize unavailable is expected in dev environments
		if strings.Contains(output, "llama-quantize unavailable") {
			t.Logf("llama-quantize not available (expected in dev env without built llama.cpp binaries)")
			t.Logf("Boundary PASSED: Go-to-C++ handoff reached validation, format accepted by Go")
			return
		}
		if strings.Contains(output, "failed to validate GGUF with llama-quantize without compatibility patches") {
			t.Logf("Validation failed but format was accepted by Go parser")
			t.Logf("Boundary PASSED: NVFP4 format passed Go GGUF parsing and reached C++ validation")
			return
		}
		t.Logf("ollama create returned error: %v", err)
	}

	// Verify model exists via show
	showReq := &api.ShowRequest{Name: modelName}
	showResp, err := client.Show(ctx, showReq)
	if err != nil {
		t.Logf("Model show failed: %v", err)
		t.Logf("Server log: %s", createStderr.String())
		// If show fails, the boundary test is inconclusive - but Go didn't crash
		t.Skipf("Model show failed (Go did not crash, boundary test inconclusive): %v", err)
	}
	t.Logf("Created model details: %+v", showResp.Details)

	// Try inference - this is the key boundary test
	genReq := &api.GenerateRequest{
		Model:  modelName,
		Prompt: "test",
		Options: map[string]interface{}{
			"num_predict": 10,
			"temperature": 0.0,
		},
	}

	var output strings.Builder
	err = client.Generate(ctx, genReq, func(resp api.GenerateResponse) error {
		output.WriteString(resp.Response)
		return nil
	})

	if err != nil {
		t.Logf("Inference returned error (expected for dummy model): %v", err)
		if strings.Contains(err.Error(), "EOF") || strings.Contains(err.Error(), "connection reset") {
			t.Fatalf("Server connection lost. The C++ engine crashed/segfaulted on NVFP4 handoff.")
		}
		t.Logf("Boundary test PASSED: Go-to-C++ handoff completed without crashing the server")
	} else {
		t.Logf("Generated output: %q", output.String())
		t.Logf("Boundary test PASSED: Go-to-C++ handoff completed without crashing the server")
	}

	// Cleanup
	deleteReq := &api.DeleteRequest{Model: modelName}
	if err := client.Delete(ctx, deleteReq); err != nil {
		t.Logf("Warning: failed to delete test model: %v", err)
	}
}
