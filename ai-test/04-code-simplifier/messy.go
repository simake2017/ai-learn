package main

import (
	"errors"
	"fmt"
	"strings"
)

// 这段代码能跑，但又臭又长，让 code-simplifier 重构

func filterAndTransform(input []string) []string {
	var result []string
	for i := 0; i < len(input); i++ {
		if input[i] != "" {
			if len(input[i]) > 0 {
				if strings.HasPrefix(input[i], "#") == false {
					upper := strings.ToUpper(input[i])
					result = append(result, upper)
				}
			}
		}
	}
	return result
}

func findItem(items []string, target string) (string, error) {
	var found string
	var foundFlag bool = false
	for idx := 0; idx < len(items); idx++ {
		if items[idx] == target {
			found = items[idx]
			foundFlag = true
			break
		}
	}
	if foundFlag == true {
		return found, nil
	} else {
		return "", errors.New("not found")
	}
}

func processWithRetry(fn func() error, maxRetries int) error {
	var lastErr error
	var retryCount int = 0
	for retryCount < maxRetries {
		err := fn()
		if err != nil {
			lastErr = err
			retryCount = retryCount + 1
			continue
		} else {
			lastErr = nil
			break
		}
	}
	if lastErr != nil {
		return fmt.Errorf("failed after %d retries: %v", retryCount, lastErr)
	}
	return nil
}

func groupByLength(words []string) map[int][]string {
	result := make(map[int][]string)
	for _, w := range words {
		l := len(w)
		if existing, ok := result[l]; ok {
			result[l] = append(existing, w)
		} else {
			result[l] = []string{w}
		}
	}
	return result
}

func main() {
	input := []string{"hello", "", "#comment", "world", "", "foo"}
	fmt.Println(filterAndTransform(input))
	fmt.Println(findItem(input, "world"))
	fmt.Println(groupByLength(input))
}
