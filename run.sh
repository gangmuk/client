#!/bin/bash

slatelog="./coefficient-and-complete-trace/Xeon-E5-2660-rs630-bg30/exclusivetime-0.1-replicated-e-c-s-trace.csv"

coefficient_file="./coefficient-and-complete-trace/Xeon-E5-2660-rs630-bg30/poly-coef_multiplied_by_one-checkoutcart.csv"

e2e_coef_file="./coefficient-and-complete-trace/c8220/e2e-poly-coef_multiplied_by_one-checkout-profile-30bg.csv"

# rps_file="./azure_dataset/request_per_min-6min.csv"
# rps_file="./azure_dataset/request_per_min-12min.csv"
# rps_file="./azure_dataset/request_per_min-30min.csv"
# rps_file="./azure_dataset/request_per_min-45min.csv" # original
rps_file="./azure_dataset/request_per_min-90min.csv" # extending 45min with smoothing

# python run_locust.py --dir_name "gangmuk-test" \
python run_test.py --dir_name "gangmuk-test" \
    --background_noise 40 \
    --degree 2 \
    --mode "runtime" \
    --routing_rule "SLATE-without-jumping" \
    --req_type "checkoutcart" \
    --slatelog ${slatelog} \
    --coefficient_file ${coefficient_file} \
    --e2e_coef_file ${e2e_coef_file} \
    --load_config 0 \
    --max_num_trace 200 \
    --load_bucket_size 100 \
    --inject_delay "[(30, 200, 'us-west-1')]" \
    --rps_file ${rps_file} \
    --duration 30 30 30 \
    --west_rps 100 300 600 \
    --east_rps 100 300 600 \
    --central_rps 100 300 600 \
    --south_rps 100 300 600

    # --background_noise delay_point noise_percentage \
    # --slatelog "exclusivetime-0.05-replicated-e-c-s-trace.csv" \
    # --inject_delay "[(timepoint, delay, region)]" \