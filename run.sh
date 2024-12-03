#!/bin/bash

dir="/users/gangmuk/projects/client/coefficient-and-complete-trace/Xeon-E5-2660-rs630-bg30/vegeta"
e2e_coef_file=${dir}/poly-rt-coef.csv
slatelog=${dir}/replicated-xt-e-c-s-trace.csv

# rps_file="./azure_dataset/request_per_min-1min-checkout.csv"
# rps_file="./azure_dataset/request_per_min-3min-checkout.csv"
# rps_file="./azure_dataset/request_per_min-6min-checkout.csv"
# rps_file="./azure_dataset/request_per_min-12min-checkout.csv"
rps_file="./azure_dataset/request_per_min-30min-checkoutcart.csv"
mode="runtime"
load_config=0
duration=300 # perform_jumping needs to be 10 second interval
dir_name="gangmuk-test-comparison"

# background_noises=(30 70)
routing_rules=("SLATE-with-jumping-global-continuous-profiling" "SLATE-with-jumping-global" "SLATE-without-jumping")
routing_rules=("SLATE-with-jumping-global-continuous-profiling")
victim_background_noise=0
background_noises=(30)
for bg in "${background_noises[@]}"; do
    for routing_rule in "${routing_rules[@]}"; do
        if [ "${routing_rule}" == "SLATE-with-jumping-global-continuous-profiling" ]; then
            coefficient_file=${dir}/poly-xt-coef-200ms.csv
            coefficient_file=${dir}/poly-xt-coef.csv
            # routing_rule="SLATE-with-jumping-global"
        else
            coefficient_file=${dir}/poly-xt-coef.csv
        fi
        echo "Running test with routing rule: ${routing_rule}"
        python run_test.py \
            --dir_name ${dir_name} \
            --background_noise "${bg}" \
            --victim_background_noise ${victim_background_noise} \
            --degree 2 \
            --mode ${mode} \
            --routing_rule "${routing_rule}" \
            --req_type "checkoutcart" \
            --slatelog "${slatelog}" \
            --coefficient_file "${coefficient_file}" \
            --e2e_coef_file "${e2e_coef_file}" \
            --load_config ${load_config} \
            --max_num_trace 200 \
            --load_bucket_size 100 \
            --inject_delay "[(1, 200, 'us-west-1')]" \
            --rps_file "${rps_file}" \
            --duration "${duration}"
        # kubectl rollout restart deployment slate-controller
        # krrd
    done
done
