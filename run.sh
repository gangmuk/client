#!/bin/bash

slatelog="./coefficient-and-complete-trace/Xeon-E5-2660-rs630-bg30/exclusivetime-0.1-replicated-e-c-s-trace.csv"

coefficient_file="./coefficient-and-complete-trace/Xeon-E5-2660-rs630-bg30/poly-coef_multiplied_by_one-checkoutcart.csv"

e2e_coef_file="./coefficient-and-complete-trace/c8220/e2e-poly-coef_multiplied_by_one-checkout-profile-30bg.csv"

rps_file="./azure_dataset/request_per_min.csv"

# python run_locust.py --dir_name "gangmuk-test" \
python run_test.py --dir_name "gangmuk-test" \
    --background_noise 30 \
    --mode "runtime" \
    --routing_rule "SLATE-with-jumping-global" \
    --req_type "checkoutcart" \
    --slatelog ${slatelog} \
    --coefficient_file ${coefficient_file} \
    --e2e_coef_file ${e2e_coef_file} \
    --load_config 0 \
    --max_num_trace 100 \
    --load_bucket_size 100 \
    --inject_delay "[(1, 1, 'us-south-1')]" \
    --rps_file ${rps_file} \
    --duration 30 30 30 \
    --west_rps 100 300 600 \
    --east_rps 100 300 600 \
    --central_rps 100 300 600 \
    --south_rps 100 300 600 \

    # --slatelog "exclusivetime-0.05-replicated-e-c-s-trace.csv" \
    # --inject_delay "[(timepoint, delay, region)]" \