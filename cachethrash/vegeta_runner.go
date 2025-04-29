package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	vegeta "github.com/tsenart/vegeta/v12/lib"
)

func parseHeaders(headerStr string) map[string][]string {
	headers := make(map[string][]string)
	if headerStr == "" {
		return headers
	}
	parts := strings.Split(headerStr, ",")
	for _, part := range parts {
		kv := strings.SplitN(part, ":", 2)
		if len(kv) == 2 {
			k := strings.TrimSpace(kv[0])
			v := strings.TrimSpace(kv[1])
			headers[k] = []string{v}
		}
	}
	return headers
}

func main() {
	// CLI flags
	method := flag.String("method", "GET", "HTTP method")
	baseURL := flag.String("url", "", "Base URL (e.g. http://host:port)")
	path := flag.String("path", "/getRecord", "Path to request (no trailing slash)")
	queryParam := flag.String("query-param", "id", "Query parameter name for ID injection")
	startID := flag.Int("start-id", 0, "Starting ID value")
	numIDs := flag.Int("num-ids", 1000, "Number of unique IDs to use before wrapping")
	headersStr := flag.String("headers", "", "Comma-separated headers (key1:val1,key2:val2)")
	rps := flag.Int("rps", 100, "Requests per second")
	duration := flag.Int("duration", 10, "Duration in seconds")
	outputBin := flag.String("outbin", "results.bin", "Output .bin file")
	outputTxt := flag.String("outtxt", "results.txt", "Output .txt file")

	flag.Parse()

	if *baseURL == "" {
		log.Fatal("Missing required flag: -url")
	}

	rate := vegeta.Rate{Freq: *rps, Per: time.Second}
	dur := time.Duration(*duration) * time.Second
	headers := parseHeaders(*headersStr)

	var counter int
	targeter := func(t *vegeta.Target) error {
		id := *startID + (counter % *numIDs)
		t.Method = *method
		t.URL = fmt.Sprintf("%s%s?%s=%d", *baseURL, *path, *queryParam, id)
		t.Header = headers
		counter++
		return nil
	}

	// Open result file
	binFile, err := os.Create(*outputBin)
	if err != nil {
		log.Fatalf("Failed to create output bin: %v", err)
	}
	defer binFile.Close()
	encoder := vegeta.NewEncoder(binFile)

	// Run attack
	attacker := vegeta.NewAttacker()
	var metrics vegeta.Metrics
	for res := range attacker.Attack(targeter, rate, dur, "query-param-test") {
		encoder.Encode(res)
		metrics.Add(res)
	}
	metrics.Close()

	// Write report
	statsFile, err := os.Create(*outputTxt)
	if err != nil {
		log.Fatalf("Failed to create output stats: %v", err)
	}
	defer statsFile.Close()

	fmt.Fprintf(statsFile, "Requests: %d\n", metrics.Requests)
	fmt.Fprintf(statsFile, "Success Rate: %.2f%%\n", metrics.Success*100)
	fmt.Fprintf(statsFile, "Avg Latency: %s\n", metrics.Latencies.Mean)
	fmt.Fprintf(statsFile, "P95 Latency: %s\n", metrics.Latencies.P95)
}
