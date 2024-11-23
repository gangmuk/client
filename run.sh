#!/bin/bash

# python run_locust.py --dir_name "gangmuk-test" \
python run_test.py --dir_name "gangmuk-test" \
    --background_noise 30 \
    --mode "runtime" \
    --routing_rule "SLATE-with-jumping-global" \
    --req_type "checkoutcart" \
    --slatelog "exclusivetime-0.1-replicated-e-c-s-trace.csv" \
    --load_config 0 \
    --max_num_trace 100 \
    --load_bucket_size 100 \
    --inject_delay "[(1, 1, 'us-south-1')]" \
    --duration 30 30 30 \
    --west_rps 100 300 600 \
    --east_rps 100 300 600 \
    --central_rps 100 300 600 \
    --south_rps 100 300 600 \

    # --slatelog "exclusivetime-0.05-replicated-e-c-s-trace.csv" \
    # --inject_delay "[(timepoint, delay, region)]" \