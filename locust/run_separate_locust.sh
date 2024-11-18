#!/bin/bash

locust -f west.py --host=http://node5.slate-gm-c6420.istio-pg0.clemson.cloudlab.us:32457 --processes 4  --master-bind-port=5557   --headless &

locust -f east.py --host=http://node5.slate-gm-c6420.istio-pg0.clemson.cloudlab.us:32457 --processes 4  --master-bind-port=5558  --headless &

locust -f south.py --host=http://node5.slate-gm-c6420.istio-pg0.clemson.cloudlab.us:32457 --processes 4  --master-bind-port=5559  --headless &

locust -f central.py --host=http://node5.slate-gm-c6420.istio-pg0.clemson.cloudlab.us:32457 --processes 4  --master-bind-port=5560 --headless  