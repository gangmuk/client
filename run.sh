#!/bin/bash

# dir="/users/gangmuk/projects/client/coefficient-and-complete-trace/Xeon-E5-2660-rs630-bg30/vegeta"
# slatelog=${dir}/replicated-xt-e-c-s-trace.csv
dir="/users/gangmuk/projects/client/coefficient-and-complete-trace/c8220/bg30"
slatelog=${dir}/replicated-e-trace-one-trace.csv
mode="runtime"
load_config=1
duration=120
dir_name="test-gangmuk"

e2e_coef_file=${dir}/poly-rt-coef.csv
coefficient_file=${dir}/poly-xt-coef.csv
victim_background_noise=(0)
delay_injected=(0)

# routing_rules=("SLATE-with-jumping-global-continuous-profiling" "SLATE-without-jumping" "SLATE-with-jumping-global" "WATERFALL2")

# routing_rules=("SLATE-with-jumping-global-with-optimizer-without-continuous-profiling")
routing_rules=("SLATE-with-jumping-global-without-optimizer-without-continuous-profiling")
rps_file_list=("rps.csv")
# cpu_limits=("checkoutservice:200m:south:130")
cpu_limits=("")
capacities=("1500")

for rps_file in "${rps_file_list[@]}"; do
    for delay in "${delay_injected[@]}"; do
        for cpu_limit in "${cpu_limits[@]}"; do
            for victim_bg in "${victim_background_noise[@]}"; do
                for routing_rule in "${routing_rules[@]}"; do
                    echo "Running test with routing rule: ${routing_rule}"
                    if [ "${routing_rule}" == "WATERFALL2" ]; then
                        for cap in "${capacities[@]}"; do
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
                                --cpu_limit "${cpu_limit}" \
                                --capacity "${cap}"
                        done
                    else
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
                            --cpu_limit "${cpu_limit}" \
                            --capacity "10000"
                    fi
                    # krrd
                done
            done
        done
    done
done
