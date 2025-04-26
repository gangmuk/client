# How to replicate cachethrash

The actual application is in `slate-benchmark/cachethrash/record_service_go`, and that has the deployment yaml and dockerfiles. A `sslateingress` is also there.

`run_cachethrash_new.py` has the actual cachethrash runner. The runner depends on `cachethrash-trace.csv` to build the call graph, `poly-coef-cachethrash.csv` and `e2e-poly-coef-cachethrash.csv` for the models. Running with degree=333 is M/M/1.

To run the sweep, run `bash sweep_cache_ratios.sh <dir>`, and you can toggle the overlap ratios there. 

For NSDI'26, the CSV for results is in `cachethrash-summary-nsdi26data.csv`. The raw data has been deleted since it is too big. To generate plots, run `python plot_sweep.py cachethrash-summary-nsdi26data.csv`. This will generate the heatmaps and the latency/weight slices.