#!/bin/bash

### three request types (checkoutcart, addtocart, emptycart)
# dir="/users/gangmuk/projects/client/coefficient-and-complete-trace/Xeon-E5-2660-rs630-bg30/vegeta/checkoutcart_addtocart_emptycart"

## two request types (checkoutcart, addtocart)
# dir="/users/gangmuk/projects/client/coefficient-and-complete-trace/Xeon-E5-2660-rs630-bg30/vegeta/checkoutcart_addtocart"

## one request type (checkoutcaft)
dir="/users/gangmuk/projects/client/coefficient-and-complete-trace/Xeon-E5-2660-rs630-bg30/vegeta/checkoutcart"

slatelog=${dir}/replicated-xt-e-c-s-trace.csv
e2e_coef_file=${dir}/poly-rt-coef.csv
coefficient_file=${dir}/poly-xt-coef.csv
mode="runtime"
load_config=1
duration=120
dir_name="test-gangmuk-full-2"
victim_background_noise=(0)
delay_injected=(0)
routing_rules=( \
    # "SLATE-without-jumping-global-with-optimizer-only-once-without-continuous-profiling" \
    # "SLATE-with-jumping-global-without-optimizer-without-continuous-profiling-init-with-optimizer" \


    # "WATERFALL2" \
    # "SLATE-without-jumping-global-without-optimizer-without-continuous-profiling-init-multi-region-routing-only-once" \
    # "SLATE-with-jumping-global-without-optimizer-without-continuous-profiling-init-with-multi-region-routing" \
    # "SLATE-without-jumping-global-with-optimizer-without-continuous-profiling" \
    # "SLATE-with-jumping-global-with-optimizer-without-continuous-profiling" \
    "SLATE-with-jumping-global-with-optimizer-with-continuous-profiling" \
)

rps_file_list=("rps.csv")
# cpu_limits=("checkoutservice:200m:south:130")
cpu_limits=("")
capacities=("1300")

for rps_file in "${rps_file_list[@]}"; do
    for delay in "${delay_injected[@]}"; do
        for cpu_limit in "${cpu_limits[@]}"; do
            for victim_bg in "${victim_background_noise[@]}"; do
                for routing_rule in "${routing_rules[@]}"; do
                    echo "Running test with routing rule: ${routing_rule}"
                    if [ "${routing_rule}" == "WATERFALL2" ]; then
                        for capacity in "${capacities[@]}"; do
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
                                --capacity "${capacity}"
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
                done
            done
        done
    done
done
