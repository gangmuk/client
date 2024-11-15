#!/bin/bash

# python run_test.py --dir_name "gangmuk-test" \
python run_locust.py --dir_name "gangmuk-locust-test" \
    --background_noise 0 \
    --mode "runtime" \
    --routing_rule "SLATE-with-jumping-global" \
    --req_type "checkoutcart" \
    --slatelog "replicated-e-c-s-trace-sample1.csv" \
    --load_config 0 \
    --max_num_trace 100 \
    --load_bucket_size 100 \
    --inject_delay "[(1, 1, 'us-south-1')]" \
    --duration 30 30 30 \
    --west_rps 100 300 600 \
    --east_rps 100 100 100 \
    --central_rps 100 100 100 \
    --south_rps 100 100 100 \

    # --duration 60 \
    # --west_rps 600 \
    # --east_rps 600 \
    # --central_rps 600 \
    # --south_rps 600 \
    
    # --inject_delay "[(1, 1, 'us-south-1')]" \
    # (timepoint, delay, region)