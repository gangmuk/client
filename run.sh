#!/bin/bash

python run_test.py --dir_name "test" \
    --background_noise 0 \
    --mode "runtime" \
    --routing_rule "SLATE-without-jumping" \
    --west_rps 100 \
    --east_rps 0 \
    --central_rps 0 \
    --south_rps 0 \
    --req_type "checkoutcart" \
    --duration 60
