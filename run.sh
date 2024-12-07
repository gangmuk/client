#!/bin/bash

dir="/users/gangmuk/projects/client/coefficient-and-complete-trace/Xeon-E5-2660-rs630-bg30/vegeta"
slatelog=${dir}/replicated-xt-e-c-s-trace.csv
mode="runtime"
load_config=0
duration=120
dir_name="dec6-continuous"

e2e_coef_file=${dir}/poly-rt-coef.csv
coefficient_file=${dir}/poly-xt-coef.csv
victim_background_noise=(0 1)
delay_injected=(0)

routing_rules=("SLATE-with-jumping-global-continuous-profiling" "SLATE-without-jumping" "SLATE-with-jumping-global")

rps_file_list=("rps.csv")
# cpu_limits=("" "checkoutcart:200m:west" "checkoutcart:200m:south")
cpu_limits=("" "checkoutcart:200m:south")


for delay in "${delay_injected[@]}"; do
    for cpu_limit in "${cpu_limits[@]}"; do
        for rps_file in "${rps_file_list[@]}"; do
            for victim_bg in "${victim_background_noise[@]}"; do
                for routing_rule in "${routing_rules[@]}"; do
                    echo "Running test with routing rule: ${routing_rule}"
                    python run_test.py \
                        --dir_name ${dir_name} \
                        --background_noise 30 \
                        --victim_background_noise ${victim_bg} \
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
                        --inject_delay "[(1, ${delay}, 'us-west-1')]" \
                        --rps_file "${rps_file}" \
                        --duration "${duration}" \
                        --cpu_limit "${cpu_limit}"
                    # kubectl rollout restart deployment slate-controller
                    # krrd
                done
            done
        done
    done
done