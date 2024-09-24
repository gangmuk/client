# Client !

## How to use

1. Configure your experiment in `config.json`
2. Run `./client` to start the client
3. Once the experiment is done, the result will be saved in `stats_output_folder` which is specified in `config.yaml`.
4. To plot the graph, run `python plot.py <stats_output_folder>`
    - Example: `python plot.py ./output-test/client0`
5. The graph will be saved in `<output_folder>/client0/result.pdf`
    - Example: `./output-test/client0/result.pdf`

If you modified the code, run `./build.sh` to build the client binary.

## Example config.yaml
config.yaml
```yaml
clients:
- headers:
    x-slate-destination: west
  method: POST
  path: /svc-a-wo-sem
  retry:
    base: 1s
    count: 0
    factor: 2
    maxinterval: 5s
  rq_timeout: 5s
  workload:
  - duration: 30s
    rps: 100
stats_output_folder: output-test
```

### Retry
Retry here is workload generator's retry logic.

The main logic is **exponential increase wait time with jitter for every retry**.

**Retry logic in main.go**
```go
waitTime := c.config.Retry.Base * time.Duration(math.Pow(float64(c.config.Retry.Factor), float64(numRetries-1)))
jitter := 0.5
jitterTime := time.Duration(rand.Float64() * jitter * float64(waitTime))
waitTime += jitterTime
waitTime = time.Duration(math.Min(float64(waitTime), float64(c.config.Retry.MaxInterval)))
```

Explanation:

- `W_base` be the base wait time (`c.config.Retry.Base`).
- `F_retry` be the retry factor (`c.config.Retry.Factor`).
- `N_retries` be the number of retries (`numRetries`).
- `J` be the jitter factor (set to 0.5).
- `W_max` be the maximum wait time (`c.config.Retry.MaxInterval`).

Steps:

1. First, calculate the initial wait time without jitter:

   `W_initial = W_base * F_retry^(N_retries - 1)`

2. Then, generate the jitter time:

   `W_jitter = J * rand() * W_initial`

   where `rand()` is a random value between 0 and 1.

3. Add the jitter time to the initial wait time:

   `W_total = W_initial + W_jitter`

4. Finally, ensure the total wait time does not exceed the maximum allowed interval:

   `W_final = min(W_total, W_max)`



### Timeout
Timeout here is workload generator's timeout logic.

The logic is simply timeout after `rq_timeout` if the request is not completed.