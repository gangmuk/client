#!/bin/bash

locust -f test2.py --host=http://node5.slate-gm-c6420.istio-pg0.clemson.cloudlab.us:32457 --processes 16
