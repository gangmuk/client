import os
import subprocess
from datetime import datetime
from kubernetes import client, config
import matplotlib.pyplot as plt
import math
import time
import paramiko
import concurrent.futures
import hashlib
import requests
import sys
import os
import concurrent.futures
from datetime import datetime, timedelta
import copy
import xml.etree.ElementTree as ET
from pprint import pprint
import numpy as np
import threading
from node_cpu import collect_cpu_utilization
import atexit
import signal
import traceback
import utils as utils
import argparse
import csv
import random
import pandas as pd
from kubernetes import client, config
# from pod_cpu import graph_pod_cpu_utilization, start_pod_cpu_monitoring
import set_cpu_limit_us_west_1
# random.seed(1234)

CLOUDLAB_CONFIG_XML="/users/gangmuk/projects/slate-benchmark/config.xml"
network_interface = "eno1"

output_dir = "./"
FULL_TEST_PATH=True

def start_node_cpu_monitoring(region_to_node, duration, filename, username="gangmuk"):
    print("Starting node CPU monitoring...")
    # Run the collect_cpu_utilization function in a separate thread
    monitoring_thread = threading.Thread(
        target=collect_cpu_utilization, args=(region_to_node, username, duration, filename)
    )
    monitoring_thread.daemon = True
    monitoring_thread.start()
    return monitoring_thread

########################################################################################
########################################################################################
# # Execute this function to clean up resources in case of a crash
def cleanup_on_crash():
    print("Cleaning up resources...")
    save_controller_logs(output_dir)
    utils.delete_tc_rule_in_client(network_interface, node_dict)
    utils.pkill_background_noise(node_dict)
    utils.pkill_stress(node_dict)
    utils.run_command(f"kubectl rollout restart deploy slate-controller")
    remove_cpu_limits_from_deployments()
    
# Signal handler
def signal_handler(signum, frame):
    print(f"Received signal: {signum}")
    # cleanup_on_crash()
    sys.exit(1)
    
# Exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    # Print exception details manually using traceback
    print("Unhandled exception:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    # Call cleanup function on crash if necessary
    # cleanup_on_crash()
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
# register this function to be called upon normal termination or unhandled exceptions. But it will not handle termination signals like SIGKILL or SIGTERM.
atexit.register(cleanup_on_crash)
sys.excepthook = handle_exception # Override the default exception hook
########################################################################################
########################################################################################
def calculate_latency_statistics(data_file, output_file):
    # Load data from the provided file
    with open(data_file, 'r') as f:
        lines = f.readlines()

    # Parse the latency values (second column) from the input data
    latencies = [float(line.split(',')[1]) for line in lines]

    # Calculate the statistics
    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    p999_latency = np.percentile(latencies, 99.9)

    # Write statistics to the output file
    with open(output_file, 'w') as f_out:
        f_out.write(f"Mean Latency: {mean_latency:.2f} ms\n")
        f_out.write(f"P50 (Median) Latency: {p50_latency:.2f} ms\n")
        f_out.write(f"P90 Latency: {p90_latency:.2f} ms\n")
        f_out.write(f"P95 Latency: {p95_latency:.2f} ms\n")
        f_out.write(f"P99 Latency: {p99_latency:.2f} ms\n")
        f_out.write(f"P999 Latency: {p999_latency:.2f} ms\n")

def set_resource_limit(resource_limit: str):
    print(f"Setting resource limit to {resource_limit} for all deployments in the default namespace...")
    # Load the kube config from default location
    config.load_kube_config()
    
    # Create a Kubernetes API client
    api_instance = client.AppsV1Api()

    # Define the resource limit in millicores (cpu)
    cpu_limit = resource_limit

    # Fetch all deployments in the default namespace
    namespace = "default"
    deployments = api_instance.list_namespaced_deployment(namespace=namespace).items

    for deployment in deployments:
        if deployment.metadata.name == "slate-controller":
            # Skip the deployment named "slate-controller"
            continue
        
        # Get the containers in the deployment spec
        containers = deployment.spec.template.spec.containers
        
        if containers:
            # Modify the first container's resource limits
            first_container = containers[0]
            
            # Check if resources field exists, if not, create one
            if not first_container.resources:
                first_container.resources = client.V1ResourceRequirements()

            if not first_container.resources.limits:
                first_container.resources.limits = {}

            # Set the CPU resource limit (convert "200mc" to proper format, e.g. "200m")
            first_container.resources.limits['cpu'] = cpu_limit
            
            # Update the deployment with the modified container resource limits
            api_instance.patch_namespaced_deployment(
                name=deployment.metadata.name,
                namespace=namespace,
                body=deployment
            )

    print("Resource limits updated for all deployments in the default namespace (except 'slate-controller').")


def remove_cpu_limits_from_deployments(namespace='default'):
    # Load the kubeconfig file (make sure you have access to the cluster)
    config.load_kube_config()

    # Initialize the API client for interacting with deployments
    v1_apps = client.AppsV1Api()

    # List all deployments in the given namespace
    deployments = v1_apps.list_namespaced_deployment(namespace=namespace)

    for deployment in deployments.items:
        updated = False  # Flag to check if we modified the deployment

        # Iterate over each container in the deployment
        for container in deployment.spec.template.spec.containers:
            # Set the resources field to None regardless of its current value
            if container.resources is not None:
                container.resources = None
                updated = True
                # print(f"Setting resources to None for container {container.name} in deployment {deployment.metadata.name}")

        # Update the deployment if we modified the resources
        if updated:
            # Use a patch to update the deployment spec
            body = {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": container.name,
                                    "resources": None
                                } for container in deployment.spec.template.spec.containers
                            ]
                        }
                    }
                }
            }

            v1_apps.patch_namespaced_deployment(
                name=deployment.metadata.name, 
                namespace=namespace, 
                body=body
            )
            # print(f"Updated deployment {deployment.metadata.name}")

def save_controller_logs(dir: str):
    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    # Define the file path
    log_file_path = os.path.join(dir, 'slate-controller.txt')

    # Run the kubectl command to get logs of the slate-controller deployment
    try:
        with open(log_file_path, 'w') as log_file:
            subprocess.run([
                'kubectl', 'logs', 'deployment/slate-controller'
            ], stdout=log_file, stderr=subprocess.PIPE, check=True)
        print(f"Logs saved to {log_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to save logs: {e.stderr.decode().strip()}")
    except FileNotFoundError:
        print("Error: kubectl command not found. Make sure it is installed and configured correctly.")

def savelogs(parentdir, services=[], regions=["us-west-1"]):
    # Directory to store the logs

    logs_directory = f"{parentdir}/proxy-logs"

    # Create the directory if it doesn't exist
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    # Get the list of all pods in the default namespace
    try:
        pod_list_output = subprocess.check_output(['kubectl', 'get', 'pods', '-n', 'default', '-o', 'jsonpath={.items[*].metadata.name}'])
        pod_list = pod_list_output.decode('utf-8').split()
        
        # Loop through each pod
        for pod_name in pod_list:
            # if pod_name STARTS WITH any of the     services, then save the logs
            if not any(pod_name.startswith(service) for service in services):
                continue

            if not any(region in pod_name for region in regions):
                continue
            # Retrieve logs of the istio-proxy container from the pod
            try:
                log_output = subprocess.check_output(['kubectl', 'logs', pod_name, '-c', 'istio-proxy', '-n', 'default'])
                
                # Save the logs to a file in the proxy-logs directory
                with open(f"{logs_directory}/{pod_name}.txt", "w") as log_file:
                    log_file.write(log_output.decode('utf-8'))
                print(f"Logs for {pod_name} saved successfully.")
            
            except subprocess.CalledProcessError as e:
                print(f"Error retrieving logs for {pod_name}: {e}")
                
    except subprocess.CalledProcessError as e:
        print(f"Error fetching pod list: s{e}")

node_dict = utils.get_nodename_and_ipaddr(CLOUDLAB_CONFIG_XML)
for node in node_dict:
    print(f"node: {node}, hostname: {node_dict[node]['hostname']}, ipaddr: {node_dict[node]['ipaddr']}")
assert len(node_dict) > 0




def update_virtualservice_latency_k8s(virtualservice_name: str, namespace: str, new_latency: str, region: str):
    config.load_kube_config()
    api = client.CustomObjectsApi()
    virtualservice = api.get_namespaced_custom_object(
        group="networking.istio.io",
        version="v1beta1",
        namespace=namespace,
        plural="virtualservices",
        name=virtualservice_name
    )
    
    print(f"update_virtualservice_latency_k8s, Trying to add {new_latency} latency in {region}, VirtualService '{virtualservice_name}'...")
    # Traverse the http routes in the spec and update the latency
    updated = False
    for http_route in virtualservice['spec'].get('http', []):
        if 'fault' in http_route and 'delay' in http_route['fault']:
            # Check if the route destination subset matches the specified region
            for route in http_route.get('route', []):
                if route['destination'].get('subset') == region:
                    # Update the fixedDelay field with the new latency value
                    http_route['fault']['delay']['fixedDelay'] = new_latency
                    updated = True

    if not updated:
        print(f"Error: update_virtualservice_latency_k8s, 2 No latency settings found in the VirtualService {region} '{virtualservice_name}' to update.")
        assert False

    # Apply the updated VirtualService back to the cluster
    api.patch_namespaced_custom_object(
        group="networking.istio.io",
        version="v1beta1",
        namespace=namespace,
        plural="virtualservices",
        name=virtualservice_name,
        body=virtualservice
    )

    print(f"update_virtualservice_latency_k8s, Updated latency to {new_latency} in VirtualService {region}, '{virtualservice_name}'.")

def update_wasm_plugin_image(namespace, plugin_name, new_image_url):
    # Load Kubernetes configuration
    config.load_kube_config()
    
    # Create an API instance to interact with CustomObjects
    api_instance = client.CustomObjectsApi()
    
    # Group, version, and plural information for the WasmPlugin
    group = 'extensions.istio.io'
    version = 'v1alpha1'
    plural = 'wasmplugins'
    
    # Fetch the current WasmPlugin configuration
    wasm_plugin = api_instance.get_namespaced_custom_object(
        group=group, version=version, namespace=namespace, plural=plural, name=plugin_name
    )
    
    # Update the image URL
    wasm_plugin['spec']['url'] = new_image_url
    
    # Patch the updated WasmPlugin back to the cluster
    api_instance.patch_namespaced_custom_object(
        group=group,
        version=version,
        namespace=namespace,
        plural=plural,
        name=plugin_name,
        body=wasm_plugin
    )
    
    print(f"Image updated to {new_image_url} in WasmPlugin '{plugin_name}'.")

def update_hillclimbing_value(namespace, plugin_name, hillclimbing_key, hillclimbing_value):
    config.load_kube_config()
    api_instance = client.CustomObjectsApi()
    group = 'extensions.istio.io'
    version = 'v1alpha1'
    plural = 'wasmplugins'

    wasm_plugin = api_instance.get_namespaced_custom_object(
        group=group, version=version, namespace=namespace, plural=plural, name=plugin_name
    )
    for env_var in wasm_plugin['spec']['vmConfig']['env']:
        if env_var['name'] == hillclimbing_key:
            env_var['value'] = hillclimbing_value
            break
    api_instance.patch_namespaced_custom_object(
        group=group,
        version=version,
        namespace=namespace,
        plural=plural,
        name=plugin_name,
        body=wasm_plugin
    )
    print(f"HILLCLIMBING key {hillclimbing_key} value updated to {hillclimbing_value} in WasmPlugin '{plugin_name}'.")


def change_jumping_mode(local=False):
    file_loc = "https://raw.githubusercontent.com/ServiceLayerNetworking/slate-wasm-bootstrap/main/slate_service.wasm"
    if local:
        file_loc = "https://raw.githubusercontent.com/ServiceLayerNetworking/slate-wasm-bootstrap/main/slate_service_local.wasm"    
    s256 = get_sha256_of_file(file_loc)
    utils.run_command(f"./change_jump_mode {s256} {file_loc}")
    if local:
        update_wasm_plugin_image("default", "slate-wasm-plugin", "ghcr.io/adiprerepa/slate-plugin:local")
    else:
        update_wasm_plugin_image("default", "slate-wasm-plugin", "ghcr.io/adiprerepa/slate-plugin:latest")


def get_sha256_of_file(url):
    # Send a GET request to the file URL
    response = requests.get(url, stream=True) 
    # Ensure the request was successful
    if response.status_code == 200:
        sha256_hash = hashlib.sha256()

        # Read the file in chunks to avoid memory overload for large files
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # Filter out keep-alive new chunks
                sha256_hash.update(chunk)

        # Return the SHA-256 hex digest
        return sha256_hash.hexdigest()
    else:
        raise Exception(f"Failed to download file: {response.status_code}")

def call_with_delay(delay, func, *args, **kwargs):
    def wrapper():
        time.sleep(int(delay))
        func(*args, **kwargs)
    t = threading.Thread(target=wrapper)
    t.daemon = True
    t.start()
    
## add to cart restart
def restart_add_to_cart():
    utils.restart_deploy(deploy=["slate-controller"], replicated_deploy=["sslateingress", "frontend", "productcatalogservice", "cartservice"], regions=["us-west-1"])
    
## checkout cart restart
def restart_checkout_cart():
    utils.restart_deploy(deploy=["slate-controller"], replicated_deploy=['currencyservice', 'emailservice', 'cartservice', 'shippingservice', 'paymentservice', 'productcatalogservice','recommendationservice','frontend','sslateingress','checkoutservice'], regions=["us-west-1"])

# Manually parse inject_delay
def parse_inject_delay(inject_delay_str):
    inject_delay_str = inject_delay_str.strip("[]")  # Remove outer brackets
    tuples = inject_delay_str.split("), (")  # Split into individual tuples

    # Clean up and convert each tuple string to actual tuple elements
    result = []
    for t in tuples:
        t = t.strip("()")  # Remove surrounding parentheses
        parts = t.split(", ")
        result.append((int(parts[0]), int(parts[1]), parts[2].strip("'\"")))
    return result

# ## smoothing rps
# def smooth_rps(df, window):
#     df["west_rps"] = df["west_rps"].rolling(window=window, min_periods=1).mean().round().astype(int)
#     df["east_rps"] = df["east_rps"].rolling(window=window, min_periods=1).mean().round().astype(int)
#     df["central_rps"] = df["central_rps"].rolling(window=window, min_periods=1).mean().round().astype(int)
#     df["south_rps"] = df["south_rps"].rolling(window=window, min_periods=1).mean().round().astype(int)
#     return df


def smooth_rps(dataframe, interval=10):
    smoothed_data = []
    for i in range(len(dataframe) - 1):
        start_row = dataframe.iloc[i]
        end_row = dataframe.iloc[i + 1]
        steps = int(start_row['duration'] // interval)
        for step in range(steps):
            t = step / steps
            new_row = {
                "west_rps": round((1 - t) * start_row['west_rps'] + t * end_row['west_rps']),
                "east_rps": round((1 - t) * start_row['east_rps'] + t * end_row['east_rps']),
                "central_rps": round((1 - t) * start_row['central_rps'] + t * end_row['central_rps']),
                "south_rps": round((1 - t) * start_row['south_rps'] + t * end_row['south_rps']),
                "request_type": start_row['request_type'],
                "duration": int(interval),
                "total_rps": round((1 - t) * start_row['total_rps'] + t * end_row['total_rps'])
            }
            smoothed_data.append(new_row)
    last_row = dataframe.iloc[-1]
    smoothed_data.append({
        "west_rps": int(last_row['west_rps']),
        "east_rps": int(last_row['east_rps']),
        "central_rps": int(last_row['central_rps']),
        "south_rps": int(last_row['south_rps']),
        "request_type": last_row['request_type'],
        "duration": int(interval),
        "total_rps": int(last_row['total_rps'])
    })
    return pd.DataFrame(smoothed_data)


def set_cpu_limit_for_a_cluster(cpu_limit, cluster):
    config.load_kube_config()
    api_instance = client.AppsV1Api()
    deployments = api_instance.list_deployment_for_all_namespaces()
    for deployment in deployments.items:
        if deployment.metadata.name.endswith(f'-us-{cluster}-1'):
            body = {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": deployment.spec.template.spec.containers[0].name,
                                    "resources": {
                                        "limits": {"cpu": cpu_limit}
                                    }
                                }
                            ]
                        }
                    }
                }
            }
            api_instance.patch_namespaced_deployment(
                name=deployment.metadata.name,
                namespace=deployment.metadata.namespace,
                body=body
            )


def remove_cpu_limits():
    config.load_kube_config()
    api_instance = client.AppsV1Api()
    deployments = api_instance.list_deployment_for_all_namespaces()
    for deployment in deployments.items:
        body = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": deployment.spec.template.spec.containers[0].name,
                                "resources": {
                                    "limits": {"cpu": None}
                                }
                            }
                        ]
                    }
                }
            }
        }
        api_instance.patch_namespaced_deployment(
            name=deployment.metadata.name,
            namespace=deployment.metadata.namespace,
            body=body
        )


def main():
    argparser = argparse.ArgumentParser(description="Run a benchmark experiment")
    argparser.add_argument("--dir_name", type=str, help="Directory name to store the experiment results", required=True)
    argparser.add_argument("--background_noise", type=int, default=0,help="Background noise level (in %)")
    argparser.add_argument("--victim_background_noise", type=int, default=0,help="Background noise level (in %)")
    argparser.add_argument("--degree", type=int, default=2, help="degree of the polynomial")
    argparser.add_argument("--mode", type=str, help="Mode of operation (profile or runtime)", required=True)
    argparser.add_argument("--routing_rule", type=str, default="SLATE-with-jumping-global", help="Routing rule to apply", choices=[\
                                "LOCAL", \
                                "SLATE-with-jumping-local", \
                                "SLATE-without-jumping", \
                                
                                "SLATE-with-jumping-global-with-optimizer-with-continuous-profiling", \
                                "SLATE-with-jumping-global-with-optimizer-without-continuous-profiling", \
                                "SLATE-without-jumping-global-with-optimizer-without-continuous-profiling", \
                                
                                "SLATE-with-jumping-global-without-optimizer-without-continuous-profiling-init-with-multi-region-routing", \
                                "SLATE-with-jumping-global-without-optimizer-without-continuous-profiling-init-with-optimizer", \
                                
                                "SLATE-without-jumping-global-with-optimizer-only-once-without-continuous-profiling", \
                                "SLATE-without-jumping-global-without-optimizer-without-continuous-profiling-init-multi-region-routing-only-once", \
                                
                                "WATERFALL2", \
                                ])
    argparser.add_argument("--duration",    type=int, default=10, required=True)
    argparser.add_argument("--req_type", type=str, default="checkoutcart", help="Request type to test")
    argparser.add_argument("--slatelog", type=str, help="Path to the slatelog file", required=True)
    argparser.add_argument("--coefficient_file", type=str, help="Path to the coefficient_file", required=True)
    argparser.add_argument("--rps_file", type=str, help="Path to the rps_file", default="")
    argparser.add_argument("--cpu_limit", type=str, help="cpu_limit", default="")
    argparser.add_argument("--e2e_coef_file", type=str, help="Path to the e2e_coef_file", required=True)
    argparser.add_argument("--load_config", type=int, default=0, help="Load coefficient flag")
    argparser.add_argument("--max_num_trace", type=int, default=0, help="max number of traces per each load bucket")
    argparser.add_argument("--load_bucket_size", type=int, default=0, help="the size of each load bucket. 'load_bucket = (rps*num_pod-(load_bucket_size//2))//(load_bucket_size) + 1'")
    argparser.add_argument("--inject_delay", type=str, help="List of tuples for injection delays (delay, time, region).")
    argparser.add_argument("--capacity", type=str, help="capacity for waterfall2")
    args = argparser.parse_args()
    
    utils.check_all_pods_are_ready()  
    
    inject_delay = parse_inject_delay(args.inject_delay)
    print("Parsed inject_delay:", inject_delay)
    if len(sys.argv) < 3:
        print("Usage: python run_test.py <dir_name>\nexit...")
        exit()
    limit = "0m"
    if len(sys.argv) == 4:
        limit = sys.argv[3]
    # if limit == "0m":
    #     remove_cpu_limits_from_deployments()
    # else:
    #     set_resource_limit(limit)
    print(f"Resource limit set to {limit} for all deployments in the default namespace.")
    CONFIG = {}
    CONFIG['background_noise'] =  args.background_noise
    CONFIG['victim_background_noise'] = args.victim_background_noise
    CONFIG['traffic_segmentation'] = 1
    '''
    # Three replicas
    # CPU: Intel(R) Xeon(R) CPU E5-2660 v2 @ 2.20GHz
    ## Based on average latency
    # checkoutcart: 800
    # addtocart: 2400
    # setcurrency: 2700
    # emptycart: 2400
    ## Based on average latency
    # checkoutcart: 700
    # addtocart: 2000
    # setcurrency: 2700
    # emptycart: 2400
    '''
    # capacity_list = [700, 1000, 1500] # assuming workload is mix of different request types
    # waterfall_capacity_set = {700, 1500} # assuming workload is mix of different request types
    # waterfall_capacity_set = {700, 1000, 1500} # assuming workload is mix of different request types
    waterfall_capacity_set = {700}
    # waterfall_capacity_set = {700, 1000}
    
    routing_rule_list = [args.routing_rule]
    onlineboutique_path = {
        "addtocart": "/cart?product_id=OLJCESPC7Z&quantity=5",
        "checkoutcart": "/cart/checkout?email=fo%40bar.com&street_address=405&zip_code=945&city=Fremont&state=CA&country=USA&credit_card_number=5555555555554444&credit_card_expiration_month=12&credit_card_expiration_year=2025&credit_card_cvv=222",
        "setcurrency": "/setCurrency?currency_code=EUR",    
        "emptycart": "/cart/empty"
    }
    experiment_list = []
    benchmark_name="onlineboutique" # a,b, 1MB and c 2MB file write
    method = "POST"
    experiment = utils.Experiment()
    if args.req_type == "checkoutcart":
        total_num_services = 8
    elif args.req_type == "addtocart":
        total_num_services = 4
    else:
        print(f"args.req_type: {args.req_type} is not supported")
        assert False
    hillclimb_interval = 30
    experiment.set_hillclimb_interval(hillclimb_interval)
    
    if not os.path.exists(args.rps_file):
        print(f"args.rps_file: {args.rps_file} does not exist")
        assert False
    # rps_df = pd.read_csv(args.rps_file, header=None, names=["request_type", "rps"])
    # if args.mode == "runtime":
    #     rps_multiplied = 5
    #     rps_df["rps"] = rps_df["rps"] * rps_multiplied
    #     rps_df["west_rps"] = rps_df["rps"].sample(frac=1, random_state=0).values
    #     rps_df["east_rps"] = rps_df["rps"].sample(frac=1, random_state=1).values
    #     rps_df["central_rps"] = rps_df["rps"].sample(frac=1, random_state=2).values
    #     rps_df["south_rps"] = rps_df["rps"].sample(frac=1, random_state=4).values
    #     rps_df["duration"] = args.duration  # Set duration
    #     rps_df.drop(columns=["rps"], inplace=True)
    # else:
    #     rps_df["west_rps"] = rps_df["rps"]
    #     rps_df["east_rps"] = 0
    #     rps_df["central_rps"] = 0
    #     rps_df["south_rps"] = 0
    #     rps_df["duration"] = args.duration
    
    # if args.mode == "runtime":
    #     unique_request_types = rps_df["request_type"].unique()
    #     new_rows = pd.DataFrame({"west_rps": 50, "east_rps":50, "central_rps":50, "south_rps":50, "request_type": unique_request_types, "duration": 60})
    #     rps_df = pd.concat([new_rows, rps_df], ignore_index=True)
        
    #     rps_df["total_rps"] = rps_df["west_rps"] + rps_df["east_rps"] + rps_df["central_rps"] + rps_df["south_rps"]
    #     max_rps = 5000
    #     for index, row in rps_df.iterrows():
    #         if row["total_rps"] > max_rps:
    #             rps_df.at[index, "west_rps"] = row["west_rps"] * max_rps / row["total_rps"]
    #             rps_df.at[index, "east_rps"] = row["east_rps"] * max_rps / row["total_rps"]
    #             rps_df.at[index, "central_rps"] = row["central_rps"] * max_rps / row["total_rps"]
    #             rps_df.at[index, "south_rps"] = row["south_rps"] * max_rps / row["total_rps"]
    #     exceeds_limit = rps_df["total_rps"] > max_rps
    #     scale_factor = np.where(exceeds_limit, max_rps / rps_df["total_rps"], 1.0)
    #     print(f"scale_factor: {scale_factor}")
    #     rps_df["west_rps"] = (rps_df["west_rps"] * scale_factor).round().astype(int)
    #     rps_df["east_rps"] = (rps_df["east_rps"] * scale_factor).round().astype(int)
    #     rps_df["central_rps"] = (rps_df["central_rps"] * scale_factor).round().astype(int)
    #     rps_df["south_rps"] = (rps_df["south_rps"] * scale_factor).round().astype(int)
    #     # rps_df = smooth_rps(rps_df, window=3)
    #     rps_df["total_rps"] = (rps_df["west_rps"] + rps_df["east_rps"] + rps_df["central_rps"] + rps_df["south_rps"])
        
    # ##############################################
    # ##### adit
    # # rps_df["west_rps"] = 300
    # # rps_df["east_rps"] = 300
    # # rps_df["central_rps"] = 300
    # # rps_df["south_rps"] = 300
    # # rps_df["duration"] = 60
    # ##############################################
    
    # set_cpu_limit_for_a_cluster("2100m", "west")
    
    rps_df = pd.read_csv(args.rps_file)
    
    # rps_df["total_rps"] = rps_df["west_rps"] + rps_df["east_rps"] + rps_df["central_rps"] + rps_df["south_rps"]
    # rps_df.to_csv("rps.csv", index=False)
    
    igw_host = utils.run_command("kubectl get nodes | grep 'node5' | awk '{print $1}'")[1]
    igw_nodeport = utils.run_command("kubectl get svc istio-ingressgateway -n istio-system -o=json | jq '.spec.ports[] | select(.name==\"http2\") | .nodePort'")[1]
    experiment_endpoint = f"http://{igw_host}:{igw_nodeport}"
    
    for request_type in rps_df["request_type"].unique():
        temp_df = rps_df[rps_df["request_type"] == request_type]
        for region in ["west", "east", "central", "south"]:
            if f"{region}_rps" in temp_df and temp_df[f"{region}_rps"].sum() > 0:
                experiment.add_workload(\
                    utils.Workload(cluster=region, \
                                    req_type=request_type, \
                                    rps=temp_df[f"{region}_rps"].to_list(), \
                                    duration=temp_df["duration"].to_list(), \
                                    method=method, path=onlineboutique_path[request_type], \
                                    endpoint=experiment_endpoint))
                print(f"Adding workload for {region} with rps: {temp_df[f'{region}_rps'].to_list()} and duration: {temp_df['duration'].to_list()}")
        # if "west_rps" in temp_df and temp_df["west_rps"].sum() > 0:
        #     experiment.add_workload(utils.Workload(cluster="west", req_type=request_type, rps=temp_df["west_rps"].to_list(), duration=temp_df["duration"].to_list(), method=method, path=onlineboutique_path[request_type], endpoint=experiment_endpoint))
        # if "east_rps" in temp_df and temp_df["east_rps"].sum() > 0:
        #     experiment.add_workload(utils.Workload(cluster="east", req_type=request_type, rps=temp_df["east_rps"].to_list(), duration=temp_df["duration"].to_list(), method=method, path=onlineboutique_path[request_type], endpoint=experiment_endpoint))
        # if "central_rps" in temp_df and temp_df["central_rps"].sum() > 0:
        #     experiment.add_workload(utils.Workload(cluster="central", req_type=request_type, rps=temp_df["central_rps"].to_list(), duration=temp_df["duration"].to_list(), method=method, path=onlineboutique_path[request_type], endpoint=experiment_endpoint))
        # if "south_rps" in temp_df and temp_df["south_rps"].sum() > 0:
        #     experiment.add_workload(utils.Workload(cluster="south", req_type=request_type, rps=temp_df["south_rps"].to_list(), duration=temp_df["duration"].to_list(), method=method, path=onlineboutique_path[request_type], endpoint=experiment_endpoint))
    
    # west_rps_str = ",".join(map(str, west_rps))
    # east_rps_str = ",".join(map(str, east_rps))
    # central_rps_str = ",".join(map(str, central_rps))
    # south_rps_str = ",".join(map(str, south_rps))
    # experiment_name = f"{args.req_type}-W{west_rps_str}-E{east_rps_str}-C{central_rps_str}-S{south_rps_str}"
    
    temp = args.rps_file.split("/")[-1].split(".")[0]
    experiment_name = f"{temp}-{','.join(rps_df['request_type'].unique())}"
    experiment.set_name(experiment_name)
    experiment_list.append(experiment)
    
    
    #### Four clusters
    region_to_node = {
        "us-west-1": ["node1"],
        "us-east-1": ["node2"],
        "us-central-1": ["node3"],
        "us-south-1": ["node4"]
    }
    region_latencies = {
        "us-west-1": {
            "us-west-1": 0,
            "us-central-1": 15,
            "us-south-1": 20,
            "us-east-1": 33,
        },
        "us-east-1": {
            "us-east-1": 0,
            "us-west-1": 33, ##### 33
            "us-south-1": 15,
            "us-central-1": 20,
        },
        "us-central-1": {
            "us-central-1": 0,
            "us-west-1": 15, ###### 15
            "us-south-1": 10,
            "us-east-1": 20,
        }, 
        "us-south-1": {
            "us-south-1": 0,
            "us-central-1": 10,
            "us-west-1": 20, ###### 20
            "us-east-1": 15,
        }
    }
    
    node_to_region = {}
    for region, nodes in region_to_node.items():
        for n in nodes:
            node_to_region[n] = region
    inter_cluster_latency = {node: {} for region in region_to_node for node in region_to_node[region]}
    for src_region, src_nodes in region_to_node.items():
        for src_node in src_nodes:
            for dst_region, dst_nodes in region_to_node.items():
                for dst_node in dst_nodes:
                    inter_cluster_latency[src_node][dst_node] = region_latencies[src_region][dst_region]
    pprint(inter_cluster_latency)
    if args.mode == "runtime":
        if FULL_TEST_PATH:
            utils.delete_tc_rule_in_client(network_interface, node_dict)
            # fault_inter_cluster_latency = copy.deepcopy(inter_cluster_latency)
            # for src, dsts in inter_cluster_latency.items():
            #     for dst in dsts:
            #         if dst == "node1": # west
            #             fault_inter_cluster_latency[src][dst] += 300 # fault in tc
            # pprint(fault_inter_cluster_latency)
            utils.apply_all_tc_rule(network_interface, inter_cluster_latency, node_dict)
    else:
        print("Skip apply_all_tc_rule in profile args.mode")
    CONFIG["mode"] = args.mode
    for src_node in inter_cluster_latency:
        for dst_node in inter_cluster_latency[src_node]:
            src_region = node_to_region[src_node]
            dst_region = node_to_region[dst_node]
            CONFIG[f"inter_cluster_latency,{src_region},{dst_region}"] = inter_cluster_latency[src_node][dst_node]
            CONFIG[f"inter_cluster_latency,{dst_region},{src_region}"] = inter_cluster_latency[src_node][dst_node]
    for experiment in experiment_list:
        for routing_rule in routing_rule_list:
            # for (point, delay, targetregion) in inject_delay:
            delay_str = f"{inject_delay[0][1]}-{inject_delay[0][2]}"
            output_dir = f"./{args.dir_name}/{experiment.name}/bg{args.background_noise}/{routing_rule}-delay-{delay_str}cap-{args.capacity}-{random.randint(0, 1000)}"
            if FULL_TEST_PATH:
                if args.background_noise > 0:
                    utils.start_background_noise(node_dict, args.background_noise, victimize_node="node1", victimize_cpu=args.background_noise)
            if args.victim_background_noise == 1:
                total_duration = rps_df["duration"].sum()
                print(f"total_duration: {total_duration}")
                call_with_delay(10, utils.run_stress, c=20, vm=10, vm_bytes=256, start_in_seconds=1, duration=total_duration+60, node_dict=node_dict, target_node="node1")
                
            print(f"mode: {args.mode}")
            print(f"routing_rule: {routing_rule}")
            utils.check_all_pods_are_ready()
            output_dir = utils.create_dir(output_dir)
            print(f"**** output_dir: {output_dir}")
            rps_df.to_csv(f"{output_dir}/workload.csv", index=False)
            for workload in experiment.workloads:
                CONFIG[f"RPS,{workload.cluster},{workload.req_type}"] = ",".join(map(str, workload.rps))
            CONFIG["benchmark_name"] = benchmark_name
            CONFIG["total_num_services"] = total_num_services
            CONFIG["degree"] = args.degree
            CONFIG["load_coef_flag"] = args.load_config
            CONFIG["routing_rule"] = routing_rule
            CONFIG["capacity"] = args.capacity
            CONFIG["hillclimb_interval"] = experiment.hillclimb_interval
            CONFIG['path'] = workload.path
            CONFIG['method'] = workload.method
            CONFIG["req_type"] = workload.req_type
            CONFIG["cluster"] = workload.cluster
            # CONFIG["duration"] = workload.duration
            CONFIG["output_dir"] = output_dir
            CONFIG["max_num_trace"] = args.max_num_trace
            CONFIG["load_bucket_size"] = args.load_bucket_size
            CONFIG["inject_delay"] = args.inject_delay
            CONFIG["cpu_limit"] = args.cpu_limit
            CONFIG["background_noise"] = args.background_noise
            CONFIG["victim_background_noise"] = args.victim_background_noise
            CONFIG["slatelog(trace)"] = args.slatelog
            CONFIG["coefficient_file"] = args.coefficient_file
            CONFIG["e2e_coef_file"] = args.e2e_coef_file
            CONFIG["load_config"] = args.load_config
            CONFIG["rps_file"] = args.rps_file
            utils.file_write_env_file(CONFIG)
            utils.file_write_config_file(CONFIG, f"{output_dir}/experiment-config.txt")
            utils.kubectl_cp_from_host_to_slate_controller_pod("env.txt", "/app/env.txt")
            if args.mode == "runtime":
                utils.kubectl_cp_from_host_to_slate_controller_pod(args.coefficient_file, "/app/coef.csv")
                utils.kubectl_cp_from_host_to_slate_controller_pod(args.e2e_coef_file, "/app/e2e-coef.csv")
            utils.kubectl_cp_from_host_to_slate_controller_pod(args.slatelog, "/app/trace.csv")
            print(f"starting experiment at {datetime.now()}, expected to finish at {datetime.now() + timedelta(seconds=sum(workload.duration))}")
            if args.mode == "runtime":
                for (point, delay, targetregion) in inject_delay:
                    if int(delay) == 0:
                        delay = 1
                    call_with_delay(point, update_virtualservice_latency_k8s, "checkoutservice-vs", "default", f"{delay}ms", targetregion)
                    print(f"update_virtualservice_latency_k8s, Delay injected: {delay}ms at {point} seconds")
            start_node_cpu_monitoring(region_to_node, sum(workload.duration), f"{output_dir}/node_cpu_util.pdf")
            if args.cpu_limit != "":
                print(f"args.cpu_limit: {args.cpu_limit}")
                try:
                    limit_list = args.cpu_limit.split(",")
                    for limit in limit_list:
                        target_deploy = limit.split(":")[0]
                        cpu_limit = limit.split(":")[1]
                        target_cluster = limit.split(":")[2]
                        delay_for_limit = int(limit.split(":")[3])
                        print(f"Going to set_cpu_limit, {target_cluster}, {target_deploy}, cpu limit: {cpu_limit} in {delay_for_limit}s")
                        call_with_delay(delay_for_limit, set_cpu_limit_us_west_1.set_cpu_limit, deploy=target_deploy, cpu_limit=cpu_limit, cluster=target_cluster)
                except Exception as e:
                    print(f"error: {e}")
                    print(f"args.cpu_limit: {args.cpu_limit} is not in the correct format")
                    print("Skip set_cpu_limit")
            else:
                print("CPU limit is empty")
                remove_cpu_limits_from_deployments()
            
            
            # start_pod_cpu_monitoring(["checkoutservice"], ["us-west-1", "us-central-1", "us-south-1", "us-east-1"], "default", sum(workload.duration), f"{output_dir}/checkout_pod_cpu_util.pdf")
            
            #####################################################################################
            #####################################################################################
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     future_list = list()
            #     for workload in experiment.workloads:
            #         future_list.append(executor.submit(utils.run_vegeta, workload, output_dir))
            #         # future_list.append(executor.submit(utils.run_newer_generation_client, workload, output_dir))
            #     for future in concurrent.futures.as_completed(future_list):
            #         print(future.result())
            # print("All clients have completed.")
            #####################################################################################
            #####################################################################################
            workloads_by_type = {}
            for workload in experiment.workloads:
                req_type = workload.req_type
                if req_type not in workloads_by_type:
                    workloads_by_type[req_type] = []
                workloads_by_type[req_type].append(workload)

            print(f"Found {len(workloads_by_type)} request types: {', '.join(workloads_by_type.keys())}")

            # Function to run all workloads of a particular request type
            def run_request_type_workloads(req_type, workloads):
                print(f"Processing request type '{req_type}' with {len(workloads)} workloads")
                
                # This assumes workloads are already in the order they should be executed
                # and that workloads for the same line (same regions) are grouped together
                
                # Group workloads by their position in the list
                # Assuming workloads from the same line are consecutive in the list
                workload_groups = []
                current_group = []
                current_duration = None
                
                for workload in workloads:
                    # If this is a new group or the duration changes, start a new group
                    if not current_group or workload.duration != current_duration:
                        if current_group:
                            workload_groups.append(current_group)
                        current_group = [workload]
                        current_duration = workload.duration
                    else:
                        current_group.append(workload)
                
                # Add the last group if it exists
                if current_group:
                    workload_groups.append(current_group)
                
                # Process each group sequentially
                results = []
                for i, group in enumerate(workload_groups):
                    print(f"Processing line group {i+1}/{len(workload_groups)} for '{req_type}' with {len(group)} regions")
                    
                    # Run all workloads in this group concurrently
                    group_futures = []
                    with concurrent.futures.ThreadPoolExecutor() as group_executor:
                        for workload in group:
                            group_future = group_executor.submit(utils.run_vegeta, workload, output_dir)
                            group_futures.append(group_future)
                        
                        # Wait for all workloads in this group to complete
                        for group_future in concurrent.futures.as_completed(group_futures):
                            result = group_future.result()
                            results.append(result)
                    
                    print(f"Completed line group {i+1}/{len(workload_groups)} for '{req_type}'")
                
                return f"Completed all workloads for request type '{req_type}'"

            # Execute different request types in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for req_type, workloads in workloads_by_type.items():
                    future = executor.submit(run_request_type_workloads, req_type, workloads)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        print(future.result())
                    except Exception as e:
                        print(f"Error executing workloads: {e}")

            print("All clients have completed.")
            #####################################################################################
            #####################################################################################
            
            utils.pkill_background_noise(node_dict)
            utils.pkill_stress(node_dict)

            flist = ["/app/endpoint_rps_history.csv", "/app/error.log"]
            for src_in_pod in flist:
                dst_in_host = f'{output_dir}/{routing_rule}-{src_in_pod.split("/")[-1]}'
                utils.kubectl_cp_from_slate_controller_to_host(src_in_pod, dst_in_host, required=False)
                # utils.run_command(f"python plot_rps.py {dst_in_host}")
            
            # print(f"output_dir: {output_dir}")
            # if os.path.exists(os.path.join(output_dir, "latency_curve")):
            #     utils.run_command(f"rm -r {output_dir}/latency_curve", required=False)
            src_directory_in_pod = "/app/poly"
            dst_directory_in_host = f"{output_dir}/latency_curve"
            os.makedirs(dst_directory_in_host, exist_ok=True)
            utils.kubectl_cp_from_slate_controller_to_host(src_directory_in_pod, dst_directory_in_host)    
            
            if args.mode == "profile":
                src_in_pod = "/app/trace_string.csv"
                dst_in_host = f"{output_dir}/trace.slatelog"
                utils.kubectl_cp_from_slate_controller_to_host(src_in_pod, dst_in_host)
                src_in_pod = "/app/global_stitched_df.csv"
                dst_in_host = f"{output_dir}/global_stitched_df.csv"
                utils.kubectl_cp_from_slate_controller_to_host(src_in_pod, dst_in_host)
                src_in_pod = "/app/coefficient.csv"
                dst_in_host = f"{output_dir}/coefficient.csv"
                utils.kubectl_cp_from_slate_controller_to_host(src_in_pod, dst_in_host)
                
            elif args.mode == "runtime":
                if "WATERFALL" in routing_rule or "SLATE" in routing_rule:
                    other_file_list = ["coefficient.csv", "routing_history.csv", "constraint.csv", "variable.csv", "network_df.csv", "compute_df.csv"]
                    for file in other_file_list:
                        src_in_pod = f"/app/{file}"
                        dst_in_host = f"{output_dir}/{routing_rule}-{file}"
                        utils.kubectl_cp_from_slate_controller_to_host(src_in_pod, dst_in_host)
                    # if routing_rule == "SLATE-with-jumping-global":
                    flist = ["jumping_routing_history.csv", "jumping_latency.csv", "region_jumping_latency.csv", "continuous_coef_dict.csv"]
                    for file in flist:
                        src_in_pod = f"/app/{file}"
                        dst_in_host = f"{output_dir}/{file}"
                        utils.kubectl_cp_from_slate_controller_to_host(src_in_pod, dst_in_host)
            else:
                print(f"args.mode: {args.mode} is not supported")
                assert False
                
            # if routing_rule.startswith("SLATE-with-jumping") and os.path.exists(f"{output_dir}/SLATE-with-jumping-global-jumping_routing_history.csv"):
            # utils.run_command(f"python {os.getcwd()}/plot_script/fast_plot.py --data_dir {output_dir}", required=False)
            utils.run_command(f"python {os.getcwd()}/plot_script/plot-vegeta.py {output_dir}", required=False)
            utils.run_command(f"python plot_script/plot_gc_jumping.py {output_dir}/jumping_routing_history.csv {output_dir}/jumping_latency.csv {output_dir}/routing_rule_plots",required=False)
            # utils.run_command(f"python plot_script/plot_region_latency.py {output_dir}/region_jumping_latency.csv {output_dir}/region_jumping_latency.pdf",required=False)
            utils.run_command(f"python plot_script/plot_endpoint_rps.py {output_dir}/{routing_rule}-endpoint_rps_history.csv {output_dir}/endpoint_rps.pdf",required=False)
            utils.run_command(f"python plot_script/plot_endpoint_rps.py {output_dir}/{routing_rule}-endpoint_rps_history.csv {output_dir}/sslateingress_endpoint_rps.pdf sslateingress",required=False)
            utils.run_command(f"python plot_script/plot_endpoint_rps.py {output_dir}/{routing_rule}-endpoint_rps_history.csv {output_dir}/frontend_endpoint_rps.pdf frontend",required=False)

            # set_cpu_limit_us_west_1.remove_cpu_limits_from_deployments()
        
            # savelogs(output_dir, services=['currencyservice', 'emailservice', 'cartservice', 'shippingservice', 'paymentservice', 'productcatalogservice','recommendationservice','frontend','sslateingress','checkoutservice'], regions=["us-central-1"])
            save_controller_logs(output_dir)
            
            time.sleep(10)
            utils.run_command(f"cp -r {output_dir} /dev/shm/")
            utils.restart_deploy(deploy=["slate-controller", \
                                        "sslateingress-us-west-1", \
                                        "sslateingress-us-east-1", \
                                        "sslateingress-us-central-1", \
                                        "sslateingress-us-south-1", \
                                        "frontend-us-west-1", \
                                        "frontend-us-east-1", \
                                        "frontend-us-central-1", \
                                        "frontend-us-south-1"])
            
            print(f"output_dir: {output_dir}")
    for node in node_dict:
        utils.run_command(f"ssh gangmuk@{node_dict[node]['hostname']} sudo tc qdisc del dev eno1 root", required=False, print_error=False)
        print(f"delete tc qdisc rule in {node_dict[node]['hostname']}")
            
if __name__ == "__main__":
    main()
