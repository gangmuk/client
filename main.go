package main

import (
	"context"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
	"golang.org/x/time/rate"
	"gopkg.in/yaml.v2"

	"client/stats"
)

type WorkloadStage struct {
	RPS      uint          `yaml:"rps"`
	Duration time.Duration `yaml:"duration"`
}

type Target struct {
	Address string `yaml:"address"`
	Port    uint   `yaml:"port"`
}

type RetryConfig struct {
	Count       int           `yaml:"count"`
	Factor      int           `yaml:"factor"`
	Base        time.Duration `yaml:"base"`
	MaxInterval time.Duration `yaml:"maxinterval"`
}

type ClientConfig struct {
	Cluster        string            `yaml:"cluster"`
	Workload       []WorkloadStage   `yaml:"workload"`
	RequestTimeout time.Duration     `yaml:"rq_timeout"`
	Retry          RetryConfig       `yaml:"retry"`
	TargetServer   Target            `yaml:"target_server"`
	Method         string            `yaml:"method"`
	Path           string            `yaml:"path"`
	Headers        map[string]string `yaml:"headers"`
}

type Config struct {
	StatsOutputFolder string         `yaml:"stats_output_folder"`
	Clients           []ClientConfig `yaml:"clients"`
}

var requestCounter uint64

type Client struct {
	config     ClientConfig
	log        *zap.SugaredLogger
	statsMgr   *stats.StatsMgr
	httpClient *http.Client
	tid        uint
	requestID  int
}

func PrintConfig(config Config) {
	fmt.Print("Client configs")
	for i, client := range config.Clients {
		fmt.Printf("====================================================")
		fmt.Printf("= Client %d:\n", i)
		fmt.Printf("= Cluster: %s\n", client.Cluster)
		fmt.Printf("= RequestTimeout in ms: %d\n", client.RequestTimeout.Milliseconds())
		fmt.Printf("= Retry Count: %d\n", client.Retry.Count)
		fmt.Printf("= Retry Factor: %d\n", client.Retry.Factor)
		fmt.Printf("= Retry Base: %s\n", client.Retry.Base)
		fmt.Printf("= Retry MaxInterval: %s\n", client.Retry.MaxInterval)
		fmt.Printf("= Target Address: %s\n", client.TargetServer.Address)
		fmt.Printf("= Target Port: %d\n", client.TargetServer.Port)
		fmt.Printf("= Method: %s\n", client.Method)
		fmt.Printf("= Path: %s\n", client.Path)
		fmt.Printf("= Headers: %v\n", client.Headers)
		for j, ws := range client.Workload {
			fmt.Printf("= Workload Stage %d:\n", j)
			fmt.Printf("= RPS: %d\n", ws.RPS)
			fmt.Printf("= Duration: %s\n", ws.Duration)
		}
		fmt.Printf("= StatsOutputFolder: %s\n", config.StatsOutputFolder)
		fmt.Printf("====================================================\n")
	}
}

func NewClient(tenantId uint, config ClientConfig, logger *zap.SugaredLogger, sm *stats.StatsMgr) *Client {
	c := Client{
		tid:      tenantId,
		config:   config,
		log:      logger,
		statsMgr: sm,
		httpClient: &http.Client{
			Timeout:   config.RequestTimeout,
			Transport: &http.Transport{},
		},
		requestID: 0,
	}
	return &c
}

func (c *Client) Start_old(wg *sync.WaitGroup, statsOutputFolder string, clientID int) {
	defer fmt.Println("Client ", clientID, " done, decrementing wg")
	defer wg.Done()
	for _, stage := range c.config.Workload {
		c.log.Infow("processing new client workload stage", "stage", stage, "client", c.tid)
		c.executeOneWorkloadStage_ratelimit(stage)
	}
	// Dump stats to folder for client after workload is done
	err := c.statsMgr.DumpStatsToFolder(statsOutputFolder + "/client-" + c.config.Cluster)
	c.log.Infow("dumping stats to folder", statsOutputFolder, "client", c.tid)
	if err != nil {
		c.log.Infof("Failed to dump client stats to folder: %v", err)
	}
	c.log.Infow("client workload finished", "client", c.tid)
}

func (c *Client) generateRequestID() string {
	return fmt.Sprintf("%d-%d", c.tid, atomic.AddUint64(&requestCounter, 1))
}

func (c *Client) sendRequest(maxRetry int, numRetries int) (a bool) {
	remainingRetry := maxRetry - numRetries
	if remainingRetry < 0 {
		return
	}

	// defer fmt.Printf("Request %d done\n", c.requestID)
	defer c.statsMgr.Incr("client.req.count", c.tid)
	targetString := fmt.Sprintf("http://%s:%d%s", c.config.TargetServer.Address, c.config.TargetServer.Port, c.config.Path)

	rqStart := time.Now()
	defer c.statsMgr.DirectMeasurement("client.req.total_hist", rqStart, 1.0, c.tid)
	req, err := http.NewRequest(c.config.Method, targetString, nil)
	if err != nil {
		c.log.Errorw("error creating request", "error", err, "client", c.tid)
		return
	}
	// c.log.Infof("sending request %s", requestID)
	req.Header.Set("X-Request-Id", strconv.Itoa(c.requestID))
	for key, value := range c.config.Headers {
		req.Header.Set(key, value)
	}
	req.Close = true
	resp, err := c.httpClient.Do(req)
	rqEnd := time.Now()
	latency := rqEnd.Sub(rqStart).Milliseconds() // Convert latency to milliseconds
	if err != nil {
		fmt.Println("Error in sending request")
		a = true
		if err, ok := err.(net.Error); ok && err.Timeout() {
			c.log.Warnw("request timed out", "client", c.tid)
			c.statsMgr.DirectMeasurement("client.req.timeout_origin", rqStart, 1.0, c.tid)
			c.statsMgr.DirectMeasurement("client.req.timeout", rqEnd, 1.0, c.tid)
			c.statsMgr.Incr("client.req.timeout.count", c.tid)
		} else {
			c.statsMgr.Incr("client.req.non_timeout_error.count", c.tid)
		}
		c.statsMgr.Incr("client.req.failure.count", c.tid)
		if remainingRetry > 0 {
			fmt.Println("Retrying request")
			// go func() {
				numRetries++
				waitTime := c.config.Retry.Base * time.Duration(math.Pow(float64(c.config.Retry.Factor), float64(numRetries-1)))
				jitter := 0.5
				jitterTime := time.Duration(rand.Float64() * jitter * float64(waitTime))
				waitTime += jitterTime
				waitTime = time.Duration(math.Min(float64(waitTime), float64(c.config.Retry.MaxInterval)))
				time.Sleep(waitTime)

				c.log.Warnw("backoff done, send retry", "requestID", c.requestID, "numRetries", numRetries, "waitTime", waitTime)
				c.statsMgr.Incr("client.req.retry.count", c.tid)
				// c.log.Infof("retrying request %s", requestID)
				c.sendRequest(maxRetry, numRetries)
			// }()
		}
		return
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		c.statsMgr.DirectMeasurement("client.req.latency", rqStart, float64(latency), c.tid)
		c.statsMgr.DirectMeasurement("client.req.success_hist", rqStart, 1.0, c.tid)
		c.statsMgr.Incr("client.req.success.count", c.tid)
	case http.StatusServiceUnavailable, http.StatusTooManyRequests:
		c.statsMgr.DirectMeasurement("client.req.503", rqStart, 1.0, c.tid)
		c.statsMgr.Incr("client.req.failure.count", c.tid)
		if remainingRetry > 0 {
			fmt.Println("Retrying request")

			// go func() {
				numRetries++
				waitTime := c.config.Retry.Base * time.Duration(math.Pow(float64(c.config.Retry.Factor), float64(numRetries-1)))
				jitter := 0.5
				jitterTime := time.Duration(rand.Float64() * jitter * float64(waitTime))
				waitTime += jitterTime
				waitTime = time.Duration(math.Min(float64(waitTime), float64(c.config.Retry.MaxInterval)))
				time.Sleep(waitTime)
				c.log.Warnw("backoff done, send retry", "requestID", c.requestID, "numRetries", numRetries, "waitTime", waitTime)
				c.statsMgr.Incr("client.req.retry.count", c.tid)
				c.sendRequest(maxRetry, numRetries)
			// }()
		}
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		c.statsMgr.DirectMeasurement("client.req.timeout_origin", rqStart, 1.0, c.tid)
		c.statsMgr.DirectMeasurement("client.req.timeout", rqEnd, 1.0, c.tid)
		c.statsMgr.Incr("client.req.failure.count", c.tid)
		if remainingRetry > 0 {
			fmt.Println("Retrying request")

			// go func() {
				numRetries++
				waitTime := c.config.Retry.Base * time.Duration(math.Pow(float64(c.config.Retry.Factor), float64(numRetries-1)))
				jitter := 0.5
				jitterTime := time.Duration(rand.Float64() * jitter * float64(waitTime))
				waitTime += jitterTime
				waitTime = time.Duration(math.Min(float64(waitTime), float64(c.config.Retry.MaxInterval)))
				time.Sleep(waitTime)
				c.log.Warnw("backoff done, send retry", "requestID", c.requestID, "numRetries", numRetries, "waitTime", waitTime)
				c.statsMgr.Incr("client.req.retry.count", c.tid)
				c.sendRequest(maxRetry, numRetries)
			// }()
		}
	default:
		c.log.Warnw("unexpected status code", "status", resp.StatusCode, "resp", resp, "client", c.tid)
		c.statsMgr.Incr("client.req.failure.count", c.tid)
	}
	return
}

// // Rate limiter implementation
func (c *Client) executeOneWorkloadStage_ratelimit(ws WorkloadStage) {

	requestSpacing := time.Second / time.Duration(ws.RPS)
	c.log.Infow("client workload stage started", "client", c.tid, "rps", ws.RPS, "duration", ws.Duration, "spacing", requestSpacing)

	// Create a rate limiter with the given RPS
	limiter := rate.NewLimiter(rate.Every(requestSpacing), 1)

	
	wg := &sync.WaitGroup{}
	counter := uint64(0)
	done := make(chan struct{})

	// Start the workload process
	wg.Add(1)
	atomic.AddUint64(&counter, 1)
	go func() {
		defer wg.Done()
		defer atomic.AddUint64(&counter, ^uint64(0))
		for {
			select {
			case <-done:
				fmt.Println("Done channel received")
				return
			default:
				// Block until we are allowed to send the next request
				err := limiter.Wait(context.Background())
				if err != nil {
					c.log.Errorw("rate limiter error", "err", err)
					return
				}

				wg.Add(1)
				atomic.AddUint64(&counter, 1)
				go func() {
					defer wg.Done()
					defer atomic.AddUint64(&counter, ^uint64(0))
					// defer fmt.Printf("Request %d done\n", c.requestID)
					numRetries := 0
					c.requestID++
					c.log.Infow("sending request", "requestID", c.requestID, "client", c.tid)
					c.statsMgr.Set("client.rps", float64(ws.RPS), c.tid)
					a := c.sendRequest(c.config.Retry.Count, numRetries)
					if a {
						fmt.Println("got failed request")
					} 	
				}()
			}
		}
	}()

	// Let the workload run for the specified duration
	time.Sleep(ws.Duration)
	fmt.Println("Waiting for requests to finish...")
	done <- struct{}{}
	c2 := make(chan struct{})
	// close(done)
	go func() {
		for {
			select {
			case <-c2:
				return
			default:
				fmt.Printf("Counter: %v\n", atomic.LoadUint64(&counter))
				time.Sleep(1 * time.Second)
			}
		}
	}()
	wg.Wait()
	close(c2)
	fmt.Println("Workload stage finished")
}

func (c *Client) executeOneWorkloadStage(ws WorkloadStage, requestWg *sync.WaitGroup) {
	requestSpacing := time.Second / time.Duration(ws.RPS)
	c.log.Infow("client workload stage started", "client", c.tid, "rps", ws.RPS, "duration", ws.Duration, "spacing", requestSpacing)
	fmt.Printf("Client %v workload stage started\n", c.tid)
	ticker := time.NewTicker(requestSpacing)

	var wg sync.WaitGroup
	done := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				wg.Add(1)        // Track individual requests
				requestWg.Add(1) // Track overall workload requests
				go func() {
					defer wg.Done()
					defer requestWg.Done() // Mark request completion
					numRetries := 0
					c.requestID++
					c.statsMgr.Set("client.rps", float64(ws.RPS), c.tid)
					c.sendRequest(0, numRetries)
				}()
			}
		}
	}()
	fmt.Printf("Waiting for %v\n", ws.Duration)
	time.Sleep(ws.Duration)
	close(done)
	ticker.Stop()
	fmt.Println("Waiting for requests to finish")
	wg.Wait()                              // Ensure all goroutines in this stage finish
	c.log.Infow("Workload stage finished") // Add log when workload stage finishes
}

func loadConfig(filename string) (Config, error) {
	var config Config
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return config, err
	}
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return config, err
	}
	return config, nil
}

// Run command helper function

func runCommand(cmd string) (string, error) {
	out, err := exec.Command("sh", "-c", cmd).Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// func periodicStatsDump(sm *stats.StatsMgr, folder string, interval time.Duration, stopChan <-chan struct{}, wg *sync.WaitGroup) {
// 	defer wg.Done()

// 	ticker := time.NewTicker(interval)
// 	defer ticker.Stop()

// 	for {
// 		select {
// 		case <-ticker.C:
// 			err := sm.DumpStatsToFolder(folder)
// 			if err != nil {
// 				fmt.Printf("Failed to dump stats to folder: %v\n", err)
// 			}
// 		case <-stopChan:
// 			return
// 		}
// 	}
// }

func main() {
	configPath := flag.String("config", "", "Path to the configuration file")
	flag.Parse()
	// Error handling for missing flag or empty string
	if *configPath == "" {
		log.Fatalf("Error: config file path is not provided. Usage: %s --config=<path-to-config.yaml>", os.Args[0])
	}
	// Check if the file exists
	if _, err := os.Stat(*configPath); os.IsNotExist(err) {
		log.Fatalf("Error: config file does not exist at path: %s", *configPath)
	}

	logger, _ := zap.NewProduction()
	defer logger.Sync()
	sugar := logger.Sugar()

	// Initial load
	appConfig, err := loadConfig(*configPath)
	if err != nil {
		log.Fatalf("error loading config: %v", err)
	}
	PrintConfig(appConfig)

	clientConfigs := appConfig.Clients
	statsOutputFolder := appConfig.StatsOutputFolder

	nodename, err := runCommand("kubectl get nodes | grep 'node3' | awk '{print $1}'")
	if err != nil {
		log.Fatalf("error getting node name: %v", err)
	}
	nodeport, err := runCommand("kubectl get svc istio-ingressgateway -n istio-system -o=json | jq '.spec.ports[] | select(.name==\"http2\") | .nodePort'")
	if err != nil {
		log.Fatalf("error getting node port: %v", err)
	}

	port, err := strconv.Atoi(nodeport)
	if err != nil {
		log.Fatalf("error converting node port to integer: %v", err)
	}

	wg := &sync.WaitGroup{}
	// var requestWg sync.WaitGroup // To track all requests globally

	clients := make([]*Client, len(clientConfigs))

	for i, clientConfig := range clientConfigs {
		clientConfig.TargetServer.Address = nodename
		clientConfig.TargetServer.Port = uint(port)

		statsMgr := stats.NewStatsMgrImpl(sugar)
		tenantId := uint(i) // or any other logic to assign tenant IDs
		c_ := NewClient(tenantId, clientConfig, sugar, statsMgr)
		clients[i] = c_

		wg.Add(1)
		go func(c_ *Client, statsMgr *stats.StatsMgr, statsOutputFolder string, clientID int) {
			defer wg.Done()

			clientWg := &sync.WaitGroup{}
			counter := uint64(0)
			clientWg.Add(1)
			atomic.AddUint64(&counter, 1)

			go c_.Start_old(clientWg, statsOutputFolder, clientID)

			c_.log.Info("Starting periodic stats collection", "client ", clientID)
			statsDone := make(chan struct{})
			c_.log.Info("Created statsDone channel", "client ", clientID)

			c_.log.Info("Adding clientWg", "client ", clientID)
			go statsMgr.PeriodicStatsCollection(1*time.Second, statsDone)
			// c_.log.Info("Started periodic stats collection", "client ", clientID)
			fmt.Println("Waiting for clientWg to finish...")
			clientWg.Wait()
			fmt.Println("clientWg finished")


			c_.log.Info("Periodic stats collection done", "client ", clientID)
			close(statsDone)
			c_.log.Info("Closed statsDone channel", "client ", clientID)

			c_.log.Info("Dumping stats to folder", "client ", clientID)
			err = statsMgr.DumpStatsToFolder(statsOutputFolder + "/client-" + c_.config.Cluster)
			if err != nil {
				sugar.Fatalw("Failed to dump client stats to folder", "client", clientID, "error", err)
			}
		}(c_, statsMgr, statsOutputFolder, i)

		logger.Sugar().Infow("started client", "client", i)
	}

	fmt.Println("Waiting for clients to finish...")
	wg.Wait() // Wait for all clients to finish
	sugar.Infow("All clients have completed. Exiting the program.")
}
