package resource

import (
	"os"
	"strconv"
	"strings"
)

const (
	// EnvTotalRAMMB overrides detected total system memory in MB. This is useful
	// when OffGrid runs inside a container whose /proc/meminfo reports the VM
	// limit instead of the host machine's memory.
	EnvTotalRAMMB = "OFFGRID_TOTAL_RAM_MB"
	// EnvAvailableRAMMB overrides detected available system memory in MB.
	EnvAvailableRAMMB = "OFFGRID_AVAILABLE_RAM_MB"
)

// ApplyMemoryOverridesMB applies memory overrides from the environment.
// If only total RAM is overridden, available RAM is estimated at 80% of total.
func ApplyMemoryOverridesMB(totalMB, availableMB int64) (int64, int64, bool) {
	overrideTotal := parseMemoryMBEnv(EnvTotalRAMMB)
	overrideAvailable := parseMemoryMBEnv(EnvAvailableRAMMB)
	changed := false

	if overrideTotal > 0 {
		totalMB = overrideTotal
		changed = true

		if overrideAvailable <= 0 {
			estimatedAvailable := overrideTotal * 80 / 100
			if estimatedAvailable > availableMB {
				availableMB = estimatedAvailable
			}
		}
	}

	if overrideAvailable > 0 {
		availableMB = overrideAvailable
		changed = true
	}

	if totalMB > 0 && availableMB > totalMB {
		availableMB = totalMB
	}

	return totalMB, availableMB, changed
}

func parseMemoryMBEnv(key string) int64 {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return 0
	}
	return parseMemoryMB(value)
}

func parseMemoryMB(value string) int64 {
	normalized := strings.ToLower(strings.TrimSpace(value))
	normalized = strings.ReplaceAll(normalized, "_", "")
	normalized = strings.ReplaceAll(normalized, " ", "")

	multiplier := int64(1)
	for _, suffix := range []string{"gib", "gb", "g"} {
		if strings.HasSuffix(normalized, suffix) {
			multiplier = 1024
			normalized = strings.TrimSuffix(normalized, suffix)
			break
		}
	}
	for _, suffix := range []string{"mib", "mb", "m"} {
		if strings.HasSuffix(normalized, suffix) {
			normalized = strings.TrimSuffix(normalized, suffix)
			break
		}
	}

	amount, err := strconv.ParseFloat(normalized, 64)
	if err != nil || amount <= 0 {
		return 0
	}

	return int64(amount * float64(multiplier))
}
