#!/bin/bash

slatelog="./coefficient-and-complete-trace/Xeon-E5-2660-rs630-bg30/exclusivetime-0.1-replicated-e-c-s-trace.csv"
coefficient_file="./coefficient-and-complete-trace/Xeon-E5-2660-rs630-bg30/poly-coef_multiplied_by_one-checkoutcart.csv"
e2e_coef_file="./coefficient-and-complete-trace/c8220/e2e-poly-coef_multiplied_by_one-checkout-profile-30bg.csv"

# slatelog="./coefficient-and-complete-trace/c6320/online-boutique.csv"
# coefficient_file="./coefficient-and-complete-trace/c6320/poly-coef_multiplied_by_one-checkout-profile-30bg.csv"
# e2e_coef_file="./coefficient-and-complete-trace/c6320/e2e-poly-coef_multiplied_by_one-checkout-profile-30bg.csv"

rps_file="./azure_dataset/request_per_min-6min-checkout.csv"
# rps_file="./azure_dataset/request_per_min-6min-checkout,addtocart.csv"
# rps_file="./azure_dataset/request_per_min-12min.csv"
# rps_file="./azure_dataset/request_per_min-30min.csv"
# rps_file="./azure_dataset/request_per_min-45min.csv" # original
# rps_file="./azure_dataset/request_per_min-90min-checkoutcart.csv" # extending 45min with smoothing
duration=60

# --background_noise delay_point noise_percentage \
# --slatelog "exclusivetime-0.05-replicated-e-c-s-trace.csv" \
# --inject_delay "[(timepoint, delay, region)]" \

# Iterating different routing rules
# list of routing rules in bash script
# routing_rules=("SLATE-with-jumping-global" "SLATE-with-jumping-global-continuous-profiling" "SLATE-without-jumping" "LOCAL")
# routing_rules=("SLATE-without-jumping" "LOCAL")
# routing_rules=("SLATE-with-jumping-global" "LOCAL")
routing_rules=("SLATE-with-jumping-global" "LOCAL" "SLATE-without-jumping")
# routing_rules=("SLATE-without-jumping")
# background_noises=(30 50 70 80)
background_noises=(30)
for routing_rule in "${routing_rules[@]}"; do
    for bg in "${background_noises[@]}"; do
        echo "Running test with routing rule: $routing_rule"
        python run_test.py \
            --dir_name "gangmuk-test-nov-29-2" \
            --background_noise "${bg}" \
            --degree 2 \
            --mode "runtime" \
            --routing_rule "${routing_rule}" \
            --req_type "checkoutcart" \
            --slatelog "${slatelog}" \
            --coefficient_file "${coefficient_file}" \
            --e2e_coef_file "${e2e_coef_file}" \
            --load_config 0 \
            --max_num_trace 200 \
            --load_bucket_size 100 \
            --inject_delay "[(1, 200, 'us-west-1')]" \
            --rps_file "${rps_file}" \
            --duration "${duration}"
    done
done