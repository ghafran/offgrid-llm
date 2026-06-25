package resource

import "testing"

func TestParseMemoryMB(t *testing.T) {
	tests := map[string]int64{
		"65536": 65536,
		"64GB":  65536,
		"64g":   65536,
		"1.5GB": 1536,
		"512MB": 512,
		"0":     0,
		"nope":  0,
	}

	for input, expected := range tests {
		if got := parseMemoryMB(input); got != expected {
			t.Fatalf("parseMemoryMB(%q) = %d, want %d", input, got, expected)
		}
	}
}

func TestApplyMemoryOverridesMB(t *testing.T) {
	t.Setenv(EnvTotalRAMMB, "64GB")

	total, available, changed := ApplyMemoryOverridesMB(8192, 4096)
	if !changed {
		t.Fatal("expected memory override to be applied")
	}
	if total != 65536 {
		t.Fatalf("total = %d, want 65536", total)
	}
	if available != 52428 {
		t.Fatalf("available = %d, want 52428", available)
	}
}

func TestApplyMemoryOverridesMBWithAvailableOverride(t *testing.T) {
	t.Setenv(EnvTotalRAMMB, "32768")
	t.Setenv(EnvAvailableRAMMB, "24576")

	total, available, changed := ApplyMemoryOverridesMB(8192, 4096)
	if !changed {
		t.Fatal("expected memory override to be applied")
	}
	if total != 32768 {
		t.Fatalf("total = %d, want 32768", total)
	}
	if available != 24576 {
		t.Fatalf("available = %d, want 24576", available)
	}
}

func TestApplyMemoryOverridesMBClampsAvailable(t *testing.T) {
	t.Setenv(EnvTotalRAMMB, "8192")
	t.Setenv(EnvAvailableRAMMB, "16384")

	total, available, changed := ApplyMemoryOverridesMB(4096, 2048)
	if !changed {
		t.Fatal("expected memory override to be applied")
	}
	if total != 8192 {
		t.Fatalf("total = %d, want 8192", total)
	}
	if available != 8192 {
		t.Fatalf("available = %d, want 8192", available)
	}
}

func TestMonitorUpdateStatsUsesMemoryOverride(t *testing.T) {
	t.Setenv(EnvTotalRAMMB, "32768")
	t.Setenv(EnvAvailableRAMMB, "24576")

	monitor := NewMonitor(0)
	monitor.updateStats()

	stats := monitor.GetStats()
	if stats.MemoryTotalMB != 32768 {
		t.Fatalf("MemoryTotalMB = %d, want 32768", stats.MemoryTotalMB)
	}
	if stats.MemoryUsedMB != 8192 {
		t.Fatalf("MemoryUsedMB = %d, want 8192", stats.MemoryUsedMB)
	}
	if stats.MemoryUsagePercent != 25 {
		t.Fatalf("MemoryUsagePercent = %f, want 25", stats.MemoryUsagePercent)
	}
}
