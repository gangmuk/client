# How to use locust

## Web Interface
1. Run it on your local laptop
```bash
ssh -L 8089:localhost:8089 gangmuk@130.127.133.224
```

2. Run locust program on the server (not on the laptop)
```bash
locust -f test2.py --host=http://node5.slate-gm-c6420.istio-pg0.clemson.cloudlab.us:32457 --processes 16
```

3. Go to the following address on your local laptop web browser
```bash
http://localhost:8080
```


## Headless
