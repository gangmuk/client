import pandas as pd
import sys
from random import sample
import logging
from threading import Lock
import config as cfg
import span as sp
import time_stitching as tst
from IPython.display import display
from pprint import pprint
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import time
import math
import matplotlib.pyplot as plt
import sys
import numpy as np
import sys
import os
import glob
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered")


latency_func = {}
is_trained_flag = False
complete_traces = {}
all_traces = {}
prerecorded_trace = {}
svc_to_rps = {}
endpoint_level_inflight = {}
endpoint_level_rps = {}
endpoint_to_cg_key = {}
ep_str_callgraph_table = {}
sp_callgraph_table = {}
all_endpoints = {}
placement = {}
coef_dict = {}
profiling = True
trace_str = list()
# x_feature = "num_inflight_dict"
x_feature = "rps_dict"
target_y = "xt" # rt

'''
cluster_to_cid and cid_to_cluster should be deprecated
cluster_id is given as a number. e.g., 0, 1, 2, ...
'''
# cluster_to_cid = {"us-west": 0, "us-east": 1}
# cid_to_cluster = {0: "us-west", 1: "us-east"}
stats_mutex = Lock()
cluster_pcts = {}


def fit_mm1_model(data, y_col_name, svc_name, ep_str, cid, directory):
    plt.figure()
    df = pd.DataFrame(data)
    x_colnames = [x for x in df.columns if x != y_col_name]
    # for x_col in x_colnames:
    if len(x_colnames) > 1:
        print(f"ERROR: {svc_name} service has more than one endpoint")
        print(f"Fit the function each endpoint separately")
        assert False
    x_col = x_colnames[0]
    df['utilization-'+x_col] = df[x_col]
    df['utilization-'+x_col] = df['utilization-'+x_col] / df['utilization-'+x_col].max()
    u_ = df['utilization-'+x_col]
    # print(f"max(u_): {u_.max()}")
    y_ = df[y_col_name]
    print(f"len(u): {len(u_)}, len(y_): {len(y_)}")
    if np.isinf(u_).any() or np.isnan(u_).any():
        print("Infinite or NaN values found in 'u'")
    print(f"u_: {u_}")
    # plt.scatter(u_, y_, color='blue', alpha=0.1, label='Data')
    max_rps = df[x_col].max()
    print(f"max_rps: {max_rps}")
    
    def mm1_model(u, a, b):
        # print(f"mm1_model, u: {u}")
        # print(f"mm1_model, len(u): {len(u)}")
        # print(f"mm1_model, a: {a}")
        # print(f"mm1_model, b: {b}")
        amplified_a = a * 1
        return ((amplified_a) / (u.max() - u)) + b
    
    
    # initial_guesses = [1, 0]  # Adjust as needed
    # bounds = (0, [np.inf, np.inf])  # Adjust as needed
    # bounds = ([0, -np.inf], [np.inf, np.inf])  # Adjust as needed
    # popt, pcov = curve_fit(mm1_model, u_, y_, p0=initial_guesses, bounds=bounds, maxfev=10000)
    
    popt, pcov = curve_fit(mm1_model, u_, y_, maxfev=10000)
    print(f"popt = {popt}")
    print(f"pcov = {pcov}")
    
    ######################################################3
    u_plot = np.linspace(min(u_), max(u_) * 0.99, 100)
    y_plot = mm1_model(u_plot, *popt)
    norm_u_ = u_*max_rps
    plt.scatter(norm_u_, y_, color='blue', alpha=0.1, label='Data')
    norm_u_plot = u_plot*max_rps
    plt.plot(norm_u_plot, y_plot, 'r-', label=f'MM1 Fit: $\\frac{{a}}{{1-u}}$, a={popt[0]}')
    plt.xlabel('Utilization (u_)')
    plt.ylabel(y_col_name + " ms")
    plt.title(f'{ep_str} in {cid}')
    plt.legend()
    plt.ylim(0, max(y_)*1.1)
    pdf_fn = f"{directory}/latency-{svc_name}-mm1-model.pdf"
    plt.savefig(pdf_fn)
    plt.show()
    print(f"Output plot saved as: {pdf_fn}")
    ######################################################3
    return {'a': popt[0]}



def fit_polynomial_regression(data, y_col_name, svc_name, ep_str, cid, directory, degree):
    # degree_list = [1,2,3,4]
    degree_list = [degree]
    plt.figure()
    df = pd.DataFrame(data)
    x_colnames = [x for x in df.columns if x != y_col_name]
    X = df[x_colnames]
    y = df[y_col_name]
    plt.scatter(X, y, color='blue', alpha=0.1, label='Data') # plot true data only once
    for degree in degree_list: # plot different degree of polynomial
        X_transformed = np.hstack((X**degree, np.ones(X.shape)))
        model = LinearRegression(fit_intercept=False)  # Intercept is manually included in X_transformed
        model.fit(X_transformed, y)
        # print(f'x_colnames: {x_colnames}')
        feature_names = x_colnames.copy() + ['intercept']
        print("model.coef_")
        print(model.coef_)
        model.coef_[0] *= 2
        coefficients = pd.Series(model.coef_, index=feature_names)

        '''plot'''
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_plot_transformed = np.hstack((X_plot**degree, np.ones(X_plot.shape)))
        y_plot = model.predict(X_plot_transformed)
        plt.plot(X_plot, y_plot, linewidth=1, label=f'Cubic Fit: $a \cdot x^{degree} + b$')
        print(f"plt.plot, degree: {degree}")
    plt.xlabel(x_feature)
    plt.ylabel(y_col_name +" ms")
    plt.title(f'{ep_str} in {cid}')
    plt.legend()
    pdf_fn = f"{directory}/latency-{x_feature}-{svc_name}.pdf"
    plt.savefig(pdf_fn)
    print(f"**output: {pdf_fn}")
    plt.show()
    return coefficients.to_dict()

def fit_and_visualize_linear_regression(data, y_col_name, svc_name, ep_str, cid):
    # Convert data to DataFrame and separate features and target
    df = pd.DataFrame(data)
    X = df.drop(columns=[y_col_name])
    y = df[y_col_name]
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Prepare coefficients and intercept for visualization and further use
    coefficients = pd.Series(model.coef_, index=X.columns).append(pd.Series([model.intercept_], index=['intercept']))
    
    # Replace negative coefficients with 1, log error
    negative_coefs = coefficients < 0
    if negative_coefs.any():
        print(coefficients[negative_coefs])
        print(f"ERROR: Negative coefficients encountered. Setting them to 1.")
        coefficients[negative_coefs] = 1
    
    # Visualize the linear regression results
    plt.scatter(X, y, color='blue', alpha=0.1)
    x_vals = pd.DataFrame([0, 30], columns=[X.columns[0]])
    y_vals = model.predict(x_vals)
    plt.plot(x_vals, y_vals, color='red', linewidth=2)
    plt.xlabel(X.columns[0])
    plt.ylabel(f'{y_col_name} (ms)')
    plt.title(f"{ep_str} in {cid}")
    
    # Save the plot
    # plt.savefig(f"latency-{X.columns[0]}-{svc_name}.pdf")
    plt.show()
    
    # Return coefficients as a dictionary
    return coefficients.to_dict()


def fit_linear_regression(data, y_col_name, svc_name, ep_str, cid):
    df = pd.DataFrame(data)
    x_colnames = list()
    for colname in df.columns:
        if colname != y_col_name:
            x_colnames.append(colname)
    X = df[x_colnames]
    y = df[y_col_name]
    model = LinearRegression()
    model.fit(X, y)
    feature_names =  list(X.columns)+ ['intercept']
    coefficients_df = pd.DataFrame(\
            {'Feature': feature_names, \
            'Coefficient':  list(model.coef_)+[model.intercept_]}\
        )
    coef = dict()
    for index, row in coefficients_df.iterrows():
        if row['Coefficient'] < 0:
            print(row)
            print(f"ERROR: row['Coefficient'] < 0: {row['Coefficient']}")
            ##########################
            row['Coefficient'] = 1
            ##########################
            # assert False
        coef[row['Feature']] = row['Coefficient']
    key_for_coef = list()
    for key in coef:
        if key == 'intercept':
            b = coef[key]
        else:
            key_for_coef.append(key)
    a = coef[key_for_coef[0]]
    x_list = [0, 30]
    y_list = list()
    for x in x_list:
        ''' linear regression '''
        y_list.append(a*x+b)
    plt.plot(X, y, 'bo', alpha=0.1)
    plt.plot(x_list, y_list, color='red', linewidth=2)
    plt.xlabel(x_feature)
    plt.ylabel(f'{target_y} (ms)')
    plt.title(ep_str + " in " + cid)
    replaced_ep_str = ep_str.replace("/", "_")
    plt.savefig(f"latency-{x_feature}-{svc_name}.pdf")
    plt.show()
    return coef


# def train_latency_function_with_trace(traces, trace_file_name, directory):
def train_latency_function_with_trace(model, traces, directory, degree):
    df = tst.trace_to_df(traces)
    coef_dict = dict()
    for cid in df["cluster_id"].unique():
        cid_df = df[df["cluster_id"]==cid]
        for svc_name in cid_df["svc_name"].unique():
            cid_svc_df = cid_df[cid_df["svc_name"]==svc_name]
            if svc_name not in latency_func:
                latency_func[svc_name] = dict()
            if svc_name not in coef_dict:
                coef_dict[svc_name] = dict()
            for ep_str in cid_svc_df["endpoint_str"].unique():
                ep_df = cid_svc_df[cid_svc_df["endpoint_str"]==ep_str]
                # Data preparation: load(X) and latency(y) 
                data = dict()
                for index, row in ep_df.iterrows():
                    flag = False
                    for key, val in row[x_feature].items(): # x_feature: rps_dict
                        # key: ep_str, val: rps
                        if key not in data:
                            data[key] = list()
                        data[key].append(val)
                        flag = True
                    if flag == True:
                        if "latency" not in data:
                            data["latency"] = list()
                        if len(row[x_feature]) == 1:
                            data["latency"].append(row[target_y])
                        else:
                            print(f"ERROR: len(row[x_feature]) != 1: {len(row[x_feature])}")
                
                ''' linear regression '''
                # coef_dict[svc_name][ep_str] = fit_linear_regression(data, y_col, svc_name, ep_str, cid)
                # coef_dict[svc_name][ep_str] = fit_and_visualize_linear_regression(data, y_col, svc_name, ep_str, cid)
                
                data = {key: value for key, value in data.items() if isinstance(value, list)}  # Ensure all values are lists
                lengths = {key: len(value) for key, value in data.items()}
                print(lengths)  # This will show you the length of the array for each column
                try:
                    df = pd.DataFrame(data)
                except ValueError as e:
                    print("Data length mismatch:", e)
                    # Optional: Log the lengths for debugging
                    print(lengths)
                    print("exit...")
                    exit()
                if model == "poly":
                    coef_dict[svc_name][ep_str] = fit_polynomial_regression(data, "latency", svc_name, ep_str, cid, directory, degree)
                elif model == "mm1":
                    coef_dict[svc_name][ep_str] = fit_mm1_model(data, "latency", svc_name, ep_str, cid, directory)
                else:
                    print(f"ERROR: model: {model}")
                    assert False
    return coef_dict


def trace_string_file_to_trace_data_structure(trace_string_file_path, required_num_svc):
    col = ["cluster_id","svc_name","method","path","trace_id","span_id","parent_span_id","st","et","rt","xt","ct","call_size","inflight_dict","rps_dict"]
    df = pd.read_csv(trace_string_file_path, names=col, header=None)
    print(f"len(df): {len(df)}")
    df = df.loc[df['rt'] > 0]
    print(f"after negative rt filter, len(df): {len(df)}")
    num_filter_rps_datapoint = 0
    list_of_span = list()
    for index, row in df.iterrows():
        if row["cluster_id"] == "SLATE_UNKNOWN_REGION" or row["svc_name"] == "consul":
            continue
        num_inflight_dict = dict()
        rps_dict = dict()
        inflight_list = row["inflight_dict"].split("|")[:-1]
        for ep_inflight in inflight_list:
            temp = ep_inflight.split(":")
            assert len(temp) == 2
            ep = temp[0]
            inflight = int(temp[1])
            num_inflight_dict[ep] = inflight
        rps_list = row["rps_dict"].split("|")[:-1]
        for ep_rps in rps_list:
            temp = ep_rps.split(":")
            assert len(temp) == 2
            ep = temp[0]
            rps = int(temp[1])
            ''' NOTE: HARDCODED, RPS FILTER'''
            if rps > 1000:
                continue
            rps_dict[ep] = rps
        ''' NOTE: HARDCODED, RPS FILTER'''
        if rps > 1200:
            num_filter_rps_datapoint += 1
            continue
        span = sp.Span(row["method"], row["path"], row["svc_name"], row["cluster_id"], row["trace_id"], row["span_id"], row["parent_span_id"], st=float(row["st"]), et=float(row["et"]), callsize=int(row["call_size"]), rps_dict=rps_dict, num_inflight_dict=num_inflight_dict)
        list_of_span.append(span)
    print(f"-- num_filter_rps_datapoint: {num_filter_rps_datapoint}")  
    all_traces = dict()
    for span in list_of_span:
        if span.cluster_id not in all_traces:
            all_traces[span.cluster_id] = dict()
        if span.trace_id not in all_traces[span.cluster_id]:
            all_traces[span.cluster_id][span.trace_id] = list()
        all_traces[span.cluster_id][span.trace_id].append(span)
    print(f"required_num_svc in {cid}: {required_num_svc}")
    complete_traces = dict()
    for cid in all_traces:
        if cid not in complete_traces:
            complete_traces[cid] = dict()
        for tid in all_traces[cid]:
            if len(all_traces[cid][tid]) == required_num_svc:
                complete_traces[cid][tid] = all_traces[cid][tid]
    for cid in all_traces:
        print(f"len(all_traces[{cid}]): {len(all_traces[cid])}")
    for cid in complete_traces:
        print(f"len(complete_traces[{cid}]): {len(complete_traces[cid])}")
    return complete_traces

def trace_string_file_to_trace_data_structure_with_df(df, required_num_svc):
    print(f"len(df): {len(df)}")
    df = df.loc[df['rt'] > 0]
    print(f"after negative rt filter, len(df): {len(df)}")
    num_filter_rps_datapoint = 0
    list_of_span = list()
    for index, row in df.iterrows():
        if row["cluster_id"] == "SLATE_UNKNOWN_REGION" or row["svc_name"] == "consul":
            continue
        num_inflight_dict = dict()
        rps_dict = dict()
        inflight_list = row["inflight_dict"].split("|")[:-1]
        for ep_inflight in inflight_list:
            temp = ep_inflight.split(":")
            assert len(temp) == 2
            ep = temp[0]
            inflight = int(temp[1])
            num_inflight_dict[ep] = inflight
        rps_list = row["rps_dict"].split("|")[:-1]
        for ep_rps in rps_list:
            temp = ep_rps.split(":")
            assert len(temp) == 2
            ep = temp[0]
            rps = int(temp[1])
            ''' NOTE: HARDCODED, RPS FILTER'''
            if rps > 1000:
                continue
            rps_dict[ep] = rps
        ''' NOTE: HARDCODED, RPS FILTER'''
        if rps > 1200:
            num_filter_rps_datapoint += 1
            continue
        span = sp.Span(row["method"], row["path"], row["svc_name"], row["cluster_id"], row["trace_id"], row["span_id"], row["parent_span_id"], st=float(row["st"]), et=float(row["et"]), callsize=int(row["call_size"]), rps_dict=rps_dict, num_inflight_dict=num_inflight_dict)
        list_of_span.append(span)
    print(f"-- num_filter_rps_datapoint: {num_filter_rps_datapoint}")  
    all_traces = dict()
    for span in list_of_span:
        if span.cluster_id not in all_traces:
            all_traces[span.cluster_id] = dict()
        if span.trace_id not in all_traces[span.cluster_id]:
            all_traces[span.cluster_id][span.trace_id] = list()
        all_traces[span.cluster_id][span.trace_id].append(span)
    print(f"required_num_svc: {required_num_svc}")
    complete_traces = dict()
    for cid in all_traces:
        if cid not in complete_traces:
            complete_traces[cid] = dict()
        for tid in all_traces[cid]:
            if len(all_traces[cid][tid]) == required_num_svc:
                complete_traces[cid][tid] = all_traces[cid][tid]
    for cid in all_traces:
        print(f"len(all_traces[{cid}]): {len(all_traces[cid])}")
    for cid in complete_traces:
        print(f"len(complete_traces[{cid}]): {len(complete_traces[cid])}")
    return complete_traces

# not used
# def training_phase(trace_file_name, directory, required_num_svc):
#     global coef_dict
#     global placement
#     global all_endpoints
#     global endpoint_to_cg_key
#     global sp_callgraph_table
#     global ep_str_callgraph_table
#     ts = time.time()
#     complete_traces = trace_string_file_to_trace_data_structure(trace_file_name, required_num_svc)
#     for cid in complete_traces:
#         print(f"len(complete_traces[{cid}]): {len(complete_traces[cid])}")
#     print(f"FILE ==> DATA STRUCTURE: {int(time.time()-ts)} seconds")
#     '''Time stitching'''
#     stitched_traces = tst.stitch_time(complete_traces)
#     # stitched_df = tst.trace_to_df(stitched_traces)
#     # print(f"len(stitched_df): {len(stitched_df)}")
#     # stitched_df = stitched_df.loc[stitched_df['rt'] > 0]
#     # stitched_df = stitched_df.loc[stitched_df['xt'] > 0]
#     # print(f"after negative xt filter, len(stitched_df): {len(stitched_df)}")
#     for cid in stitched_traces:
#         print(f"len(stitched_traces[{cid}]): {len(stitched_traces[cid])}")
#     '''Create useful data structures from the traces'''
#     sp_callgraph_table = tst.traces_to_span_callgraph_table(stitched_traces)
#     endpoint_to_cg_key = tst.get_endpoint_to_cg_key_map(stitched_traces)
#     ep_str_callgraph_table = tst.traces_to_endpoint_str_callgraph_table(stitched_traces)
#     print("ep_str_callgraph_table")
#     print(f"num different callgraph: {len(ep_str_callgraph_table)}")
#     for cg_key in ep_str_callgraph_table:
#         print(f"{cg_key}: {ep_str_callgraph_table[cg_key]}")
#     all_endpoints = tst.get_all_endpoints(stitched_traces)
#     if cfg.OUTPUT_WRITE:
#         tst.file_write_callgraph_table(sp_callgraph_table)
#     placement = tst.get_placement_from_trace(stitched_traces)
#     for cid in placement:
#         print(f"placement[{cid}]: {placement[cid]}")
#     poly_coef_dict, mm1_coef_dict = train_latency_function_with_trace(stitched_traces, directory)
#     print("-"*60)
#     print("poly_coef_dict before checking")
#     for svc_name in poly_coef_dict:
#         for ep_str in poly_coef_dict[svc_name]:
#             print(f'poly_coef_dict[{svc_name}][{ep_str}]: {poly_coef_dict[svc_name][ep_str]}')
#     print("-"*60)
#     print("mm1_coef_dict")
#     pprint(mm1_coef_dict)
#     print("-"*60)
#     # NOTE: latency function should be strictly increasing function
#     ''' linear regression '''
#     for svc_name in coef_dict: # svc_name: metrics-db
#         for ep_str in coef_dict[svc_name]: # ep_str: metrics-db@GET@/dbcall
#             for feature_ep in coef_dict[svc_name][ep_str]: # feature_ep: 'metrics-db@GET@/dbcall' or 'intercept'
#                 if feature_ep != "intercept": # a in a*(x^degree) + b
#                     if coef_dict[svc_name][ep_str][feature_ep] < 0:
#                         coef_dict[svc_name][ep_str][feature_ep] = 0
#                         # coef_dict[svc_name][ep_str]['intercept'] = 1
#                         print(f"WARNING!!!: coef_dict[{svc_name}][{ep_str}] coefficient is negative. Set it to 0.")
#                     else: 
#                         if coef_dict[svc_name][ep_str]['intercept'] < 0:
#                             # a is positive but intercept is negative
#                             coef_dict[svc_name][ep_str]['intercept'] = 1
#                             print(f"WARNING: coef_dict[{svc_name}][{ep_str}], coefficient is positive.")
#                             print(f"WARNING: But, coef_dict[{svc_name}][{ep_str}], intercept is negative. Set it to 0.")
#     ''' MM1 model '''
#     for svc_name in coef_dict: # svc_name: metrics-db
#         for ep_str in coef_dict[svc_name]:
#             for feature_ep in coef_dict[svc_name][ep_str]:
#                 if feature_ep != "intercept":
#                     if coef_dict[svc_name][ep_str][feature_ep] < 0:
#                         coef_dict[svc_name][ep_str][feature_ep] = 0
#                         print(f"WARNING!!!: coef_dict[{svc_name}][{ep_str}] coefficient is negative. Set it to 0.")
#                     else: 
#                         if coef_dict[svc_name][ep_str]['intercept'] < 0:
#                             coef_dict[svc_name][ep_str]['intercept'] = 1
#                             print(f"WARNING: But, coef_dict[{svc_name}][{ep_str}], intercept is negative. Set it to 0.")
#     print("coef_dict after checking")
#     for svc_name in coef_dict:
#         for ep_str in coef_dict[svc_name]:
#             print(f'coef_dict[{svc_name}][{ep_str}]: {coef_dict[svc_name][ep_str]}')


def merge_files(directory, postfix):
    slatelog_files = glob.glob(os.path.join(directory, '**', f'*{postfix}'), recursive=True)
    output_filename = f"merged{postfix}"
    with open(output_filename, 'w') as outfile:
        for fname in slatelog_files:
            with open(fname) as infile:
                outfile.write(infile.read())
    return output_filename


if __name__ == "__main__":
    directory = sys.argv[1]
    required_num_svc= int(sys.argv[2])
    if len(sys.argv) < 2:
        print("Usage: python3 replicate.py <directory> <required_num_svc>")
        sys.exit(1)
    merged_trace_file_name = merge_files(directory, ".slatelog")
    print(f"merged_trace_file_name: {merged_trace_file_name}")
    columns = ["cluster_id","svc_name","method","path","trace_id","span_id","parent_span_id","st","et","rt","xt","ct","call_size","inflight_dict","rps_dict"]
    df = pd.read_csv(merged_trace_file_name, header=None, names=columns)
    trace_span_counts = df.groupby('trace_id').size()
    trace_ids_with_four_spans = trace_span_counts[trace_span_counts == required_num_svc].index
    filtered_df = df[df['trace_id'].isin(trace_ids_with_four_spans)]
    trace_id = filtered_df['trace_id'].unique().tolist()
    sample_ratio = 1.0
    sample_size = int(len(trace_id) * sample_ratio)
    sampled_trace_id = sample(trace_id, sample_size)
    double_filtered_df = filtered_df[filtered_df['trace_id'].isin(sampled_trace_id)]
    # training_phase(merged_trace_file_name, directory, required_num_svc)
    service_list = df['svc_name'].unique().tolist()
    print(f"service_list: {service_list}")
    complete_traces = trace_string_file_to_trace_data_structure_with_df(double_filtered_df, required_num_svc)
    for cid in complete_traces:
        print(f"len(complete_traces[{cid}]): {len(complete_traces[cid])}")
    stitched_traces = tst.stitch_time(complete_traces)
    for cid in stitched_traces:
        print(f"len(stitched_traces[{cid}]): {len(stitched_traces[cid])}")
        
    mm1_coef_dict = train_latency_function_with_trace("mm1", stitched_traces, directory, degree=None)
    with open(f"/users/gangmuk/projects/DeathStarBench/hotelReservation/coef_multiplied_by_two.csv", "a") as f:
        for svc_name in mm1_coef_dict:
            for endpoint in mm1_coef_dict[svc_name]:
                for feature in mm1_coef_dict[svc_name][endpoint]:
                    print(f"mm1_coef_dict,{svc_name},{endpoint},{feature},{mm1_coef_dict[svc_name][endpoint][feature]}")
    exit()
    
    degree = 2 # NOTE
    poly_coef_dict = train_latency_function_with_trace("poly", stitched_traces, directory, degree)
    print("-"*80)
    # with open(f"{directory}/poly_coef_dict.csv", "w") as f:
    with open(f"/users/gangmuk/projects/DeathStarBench/hotelReservation/coef_multiplied_by_two.csv", "a") as f:
        for svc_name in poly_coef_dict:
            for ep_str in poly_coef_dict[svc_name]:
                for feature in poly_coef_dict[svc_name][ep_str]:
                    print(f'poly_coef_dict,{svc_name},{ep_str},{feature},{poly_coef_dict[svc_name][ep_str][feature]}')
                    f.write(f'{svc_name},{ep_str},{feature},{poly_coef_dict[svc_name][ep_str][feature]}\n')
                
    print("-"*80)
    
    print("num all trace ", len(df['trace_id'].unique()))
    print("num complete trace", len(filtered_df['trace_id'].unique()))
    print("num sampled trace", len(double_filtered_df['trace_id'].unique()))
    ''' Define how you want to replicate '''
    new_cluster_dict = dict()
    # replicated_cluster_list = ["us-east-1", "us-central-1", "us-south-1"]
    replicated_cluster_list = []
    for cluster in replicated_cluster_list:
        new_cluster_dict[cluster] = service_list
    new_df_dict = dict()
    for nc in new_cluster_dict:
        copy_df = double_filtered_df.copy() # copy orignal trace log df
        copy_df['cluster_id'] = nc # set 'cluster_id' column to a new cluster name
        copy_df = copy_df[copy_df['svc_name'].isin(new_cluster_dict[nc])] # filter out services that you don't want to replicate
        new_df_dict[nc] = copy_df.copy()
        print(f"Replicated {nc}")
    df_all = double_filtered_df.copy()
    for cluster_id, new_df in new_df_dict.items():
        df_all = pd.concat([df_all, new_df])
    df_all.sort_values(by=['cluster_id', 'trace_id'], inplace=True)
    output_fn = "replicated-"
    for nc in new_cluster_dict:
        cluster_id_first_ch = nc.split('-')[1][0]
        output_fn += f"{cluster_id_first_ch}-"
    output_fn += "trace.csv"
    output_path = directory + output_fn
    df_all.to_csv(output_path, index=False, header=False)
    print("Output file written: ", output_path)
