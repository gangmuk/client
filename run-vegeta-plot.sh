#!/bin/bash

dir="./dec-2-endtoend/request_per_min-30min-checkoutcart-checkoutcart/bg30/finish"


# python plot_script/plot-vegeta.py ${dir}/LOCAL/

python plot_script/plot-vegeta.py ${dir}/SLATE-without-jumping/


python plot_script/plot-vegeta.py ${dir}/SLATE-with-jumping-global/


python plot_script/plot-vegeta.py ${dir}/SLATE-with-jumping-global-continuous-profiling/
