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
import atexit
import signal
import traceback
import utils

# Get the parent directory (client)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from node_cpu import collect_cpu_utilization
from pod_cpu import graph_pod_cpu_utilization
import egress

CLOUDLAB_CONFIG_XML="/users/gangmuk/projects/slate-benchmark/config.xml"
network_interface = "eno1"
FAST_TEST = False

def start_node_cpu_monitoring(region_to_node, duration, filename, username="gangmuk"):
    # Run the collect_cpu_utilization function in a separate thread
    monitoring_thread = threading.Thread(
        target=collect_cpu_utilization, args=(region_to_node, username, duration, filename)
    )
    monitoring_thread.daemon = True
    monitoring_thread.start()
    return monitoring_thread

def start_pod_cpu_monitoring(deployments, regions, namespace, duration, filename):
    # Run the graph_pod_cpu_utilization function in a separate thread
    monitoring_thread = threading.Thread(
        target=graph_pod_cpu_utilization, args=(deployments, regions, namespace, duration, filename)
    )
    monitoring_thread.daemon = True
    monitoring_thread.start()
    return monitoring_thread
########################################################################################
########################################################################################
# # Execute this function to clean up resources in case of a crash
def cleanup_on_crash():
    print("Cleaning up resources...")
    utils.restart_deploy(deploy=["slate-controller"])
    utils.delete_tc_rule_in_client(network_interface, node_dict)
    utils.pkill_background_noise(node_dict)
    
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
                print(f"Setting resources to None for container {container.name} in deployment {deployment.metadata.name}")

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
            print(f"Updated deployment {deployment.metadata.name}")

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
    # Load the Kubernetes config (make sure kubeconfig is set)
    config.load_kube_config()

    # Create an API client for CustomObjects (used for interacting with CRDs like VirtualService)
    api = client.CustomObjectsApi()

    # Get the existing VirtualService
    virtualservice = api.get_namespaced_custom_object(
        group="networking.istio.io",
        version="v1beta1",
        namespace=namespace,
        plural="virtualservices",
        name=virtualservice_name
    )

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
        print(f"No latency settings found in the VirtualService '{virtualservice_name}' to update.")
        return

    # Apply the updated VirtualService back to the cluster
    api.patch_namespaced_custom_object(
        group="networking.istio.io",
        version="v1beta1",
        namespace=namespace,
        plural="virtualservices",
        name=virtualservice_name,
        body=virtualservice
    )

    print(f"Updated latency to {new_latency} in VirtualService '{virtualservice_name}'.")

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

from kubernetes import client, config

def set_env_vars_in_deployment(deployment_name, namespace, env_vars: dict):
    """
    Set or update multiple environment variables in a Kubernetes deployment.

    :param deployment_name: Name of the deployment
    :param namespace: Kubernetes namespace
    :param env_vars: Dictionary of environment variables to set {key: value}
    """
    # Load Kubernetes config (from default kubeconfig or in-cluster config)
    config.load_kube_config()

    api = client.AppsV1Api()

    # Get current deployment
    deployment = api.read_namespaced_deployment(name=deployment_name, namespace=namespace)

    for container in deployment.spec.template.spec.containers:
        if container.env is None:
            container.env = []

        # Create a map of current env vars for quick lookup
        current_env = {env.name: env for env in container.env}

        for key, value in env_vars.items():
            if key in current_env:
                current_env[key].value = value
            else:
                container.env.append(client.V1EnvVar(name=key, value=value))

    # Patch the deployment
    api.patch_namespaced_deployment(
        name=deployment_name,
        namespace=namespace,
        body=deployment
    )

    print(f"Updated env vars {list(env_vars.keys())} in deployment {deployment_name}")


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

def enrich_normalization(norm_dict):
    """
    Enrich the normalization dict with all the inverses.
    """
    for key in list(norm_dict.keys()):
        for subkey in list(norm_dict[key].keys()):
            # Check if the value under norm_dict[key][subkey] is a number
            if isinstance(norm_dict[key][subkey], (int, float)):
                # Add inverse to the subkey if not already present
                if subkey not in norm_dict:
                    norm_dict[subkey] = dict()
                # Create the inverse
                norm_dict[subkey][key] = 1 / norm_dict[key][subkey]
    return norm_dict

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
        time.sleep(delay)
        func(*args, **kwargs)
    t = threading.Thread(target=wrapper)
    t.daemon = True
    t.start()

def main():
    background_noise = int(sys.argv[1])
    sys_arg_dir_name = sys.argv[2]
    if len(sys.argv) < 3:
        print("Usage: python run_wrk.py <dir_name>\nexit...")
        exit()
    
    westCacheStart = 0
    southCacheStart = 100
    if len(sys.argv) == 5:
        westCacheStart = int(sys.argv[3])
        southCacheStart = int(sys.argv[4])
    
    CONFIG = {}
    CONFIG['background_noise'] =  background_noise
    CONFIG['traffic_segmentation'] = 1
    igw_host = utils.run_command("kubectl get nodes | grep 'node5' | awk '{print $1}'")[1]
    igw_nodeport = utils.run_command("kubectl get svc istio-ingressgateway -n istio-system -o=json | jq '.spec.ports[] | select(.name==\"http2\") | .nodePort'")[1]
    experiment_endpoint = f"http://{igw_host}:{igw_nodeport}"

    degree = 2
    # degree = 111 # mm1
    # degree = 222 # mm1 approximated with piecewise linear function
    # degree = 333 # mm1 approximated with piecewise linear function and also with load shedding variable
    
    # mode = "profile"
    mode = "runtime"
    # routing_rule_list = ["LOCAL"]
    # routing_rule_list = ["SLATE-without-jumping", "SLATE-with-jumping-global", "SLATE-with-jumping-local"]
    # routing_rule_list = ["SLATE-without-jumping"]
    # routing_rule_list = ["SLATE-with-jumping-global"]
    routing_rule_list = [ \
                        # "SLWTE-with-jumping-global", \
                        # "SLATE-without-jumping", \
                            
                        # "SLATE-without-jumping-global-with-optimizer-without-continuous-profiling", \
                        # "SLATE-without-jumping-global-with-optimizer-with-continuous-profiling", \
                        "SLATE-with-jumping-global-with-optimizer-without-continuous-profiling", \
                        # "SLATE-with-jumping-global-with-optimizer-with-continuous-profiling", \
                        ]

    # routing_rule_list = ["WATERFALL2"]
    
    onlineboutique_path = {
        "getRecord": "/getRecord"
    }
    
    ####################################################################
    ###################### Workload ####################################
    ####################################################################
    experiment_list = []
    
    benchmark_name="cachethrash"
    method = "GET"
    experiment = utils.Experiment()
    total_num_services = 2
 
    workloads = {
        "w100c500e1200s100": {
            "west": {
                # "checkoutcart": [(0, 50), (60, 400), (360, 1200)],
                "getRecord": [(0, 100)],
                # "addtocart": [(0, 500)],
                # "emptycart": [(0, 100)]
            },
            "central": {
                # "checkoutcart": [(0, 50), (60, 800), (360, 700)],
                "getRecord": [(0, 1150)],
                # "addtocart": [(0, 2500)],
                # "emptycart": [(0, 1)]
            },
            "east": {
            #     "checkoutcart": [(0, 50), (60, 1300), (360, 800)],
                "getRecord": [(0, 300)],

            # #     "multicore": [(0, 200)],
            },
            "south": {
            #     "checkoutcart": [(0, 50), (60, 600), (360, 500)],
                "getRecord": [(0, 100)],


            # #     "multicore": [(0, 200)],
            },
        }
    }
    cachespace = {
        "west": {
            "getRecord": (westCacheStart, 100),
        },
        "central": {
            "getRecord": (0, 100),
        },
        "east": {
            "getRecord": (200, 100),
        },
        "south": {
            "getRecord": (southCacheStart, 100),
        },
    }
    vegeta_per = {
        # "central": {
        #     "emptycart": "30s"
        # }
    }

    # Profiling workload
    # workloads = {}
    
    # for rps in range(800, 1001, 50):  # Start at 100, end at 800, step by 100
    #     key = f"w{rps}"
    #     workloads[key] = {
    #         "west": {
    #             "getRecord": [(0, rps)]
    #         }
    #     }

    
    

    
    def construct_dur_list(workload_list, experiment_length):
        """
        construct_dur_list will take a list of tuples (start, rps) and return a list of durations for those respective rps.
        the list should sum to experiment_length
        """
        dur_list = []
        for i in range(len(workload_list)):
            start, rps = workload_list[i]
            if i == len(workload_list) - 1:
                dur_list.append(experiment_length - start)
            else:
                dur_list.append(workload_list[i+1][0] - start)
        return dur_list

    for name, w_ in workloads.items():
        jumping_interval = 15
        jumping_warm_start = 2

        experiment_dur = 60*3
        # actual normalization: 4
        # CPU-based normalization: 1/4
        normalization_dict = {
            # "sslateingress@POST@/cart/checkout": {
            #     "sslateingress@POST@/cart": 1, # 3.12
            # },
            # "frontend@POST@/cart/checkout": {
            #     "frontend@POST@/cart": 1,
            # },
            # "cartservice@POST@/hipstershop.CartService/GetCart": {
            #     "cartservice@POST@/hipstershop.CartService/AddItem": 10,
            # },
        }
        normalization_dict = enrich_normalization(normalization_dict)
        
        experiment = utils.Experiment()
        
        experiment.normalization = normalization_dict
        experiment.workload_raw = w_
        experiment.jumping_interval = jumping_interval
        experiment.jumping_warm_start = jumping_warm_start
        for region in w_:
            for req_type in w_[region]:
                # rps_list is the second element of the tuple in the list
                rps_list = [rps for start, rps in w_[region][req_type]]
                workload = utils.Workload(cluster=region, req_type=req_type, rps=rps_list, duration=construct_dur_list(w_[region][req_type], experiment_dur),
                                                        method=method, path=onlineboutique_path[req_type], hdrs={}, endpoint=experiment_endpoint, vegeta_per=vegeta_per.get(region, {}).get(req_type, "1s"))
                workload.start_id = cachespace[region][req_type][0]
                workload.num_ids = cachespace[region][req_type][1]
                experiment.add_workload(workload)
        experiment.hash_mod = 5
        experiment_name = f"{req_type}-{name}"
        # experiment.set_injected_delay([(1, 300, "us-west-1")])
        experiment.set_name(experiment_name)
        experiment_list.append(experiment)
    ####################################################################
    ####################################################################
    
    
    #####################################
    #### Four clusters
    region_to_node = {
        "us-west-1": ["node1"],
        "us-east-1": ["node2"],
        "us-central-1": ["node3"],
        "us-south-1": ["node4"]
    }
    
    region_latencies = {
        "us-west-1": {
            "us-central-1": 30,
            "us-south-1": 200,
            "us-east-1": 330,
        },
        "us-east-1": {
            "us-south-1": 150,
            "us-central-1": 200,
        },
        "us-central-1": {
            "us-south-1": 30,
        }
    }
    # region_latencies = {
    #     "us-west-1": {
    #         "us-central-1": 15,
    #         "us-south-1": 20,
    #         "us-east-1": 33,
    #     },
    #     "us-east-1": {
    #         "us-south-1": 15,
    #         "us-central-1": 20,
    #     },
    #     "us-central-1": {
    #         "us-south-1": 10,
    #     }
    # }
    
    node_to_region = {}
    for region, nodes in region_to_node.items():
        for n in nodes:
            node_to_region[n] = region
    
    # GCP, OR, SLC, IOW, SC
    # Collect all unique regions
    all_regions = set(region_latencies.keys())
    for region in region_latencies:
        all_regions.update(region_latencies[region].keys())
    # ensure symmetry
    for region in all_regions:
        if region not in region_latencies:
            region_latencies[region] = {}
        for other_region in all_regions:
            if other_region not in region_latencies[region]:
                if region == other_region:
                    region_latencies[region][other_region] = 0  # latency to self is zero
                elif other_region in region_latencies and region in region_latencies[other_region]:
                    # Copy the reverse latency if it exists
                    region_latencies[region][other_region] = region_latencies[other_region][region]
                else:
                    # Initialize to None if no data is available in either direction
                    region_latencies[region][other_region] = None
    
    inter_cluster_latency = {node: {} for region in region_to_node for node in region_to_node[region]}
    for src_region, src_nodes in region_to_node.items():
        for src_node in src_nodes:
            for dst_region, dst_nodes in region_to_node.items():
                for dst_node in dst_nodes:
                    inter_cluster_latency[src_node][dst_node] = region_latencies[src_region][dst_region]
    print("inter_cluster_latency")
    pprint(inter_cluster_latency)
    
    if not FAST_TEST:
        utils.pkill_background_noise(node_dict)


    if mode == "runtime":
        if not FAST_TEST:
            utils.delete_tc_rule_in_client(network_interface, node_dict)
            utils.apply_all_tc_rule(network_interface, inter_cluster_latency, node_dict)
    else:
        print("Skip apply_all_tc_rule in profile mode")

    CONFIG["mode"] = mode
    for src_node in inter_cluster_latency:
        for dst_node in inter_cluster_latency[src_node]:
            src_region = node_to_region[src_node]
            dst_region = node_to_region[dst_node]
            CONFIG[f"inter_cluster_latency,{src_region},{dst_region}"] = inter_cluster_latency[src_node][dst_node]
            CONFIG[f"inter_cluster_latency,{dst_region},{src_region}"] = inter_cluster_latency[src_node][dst_node]
    CONFIG["benchmark_name"] = benchmark_name
    CONFIG["total_num_services"] = total_num_services
    CONFIG["degree"] = degree
    CONFIG["load_coef_flag"] = 1
    
    
    # utils.restart_deploy(deploy=["slate-controller"])
    
    for experiment in experiment_list:
        for routing_rule in routing_rule_list:
            if not FAST_TEST:
                utils.start_background_noise(node_dict, CONFIG['background_noise'], victimize_node="node1", victimize_cpu=CONFIG['background_noise'])

            # update_virtualservice_latency_k8s("checkoutservice-vs", "default", f"1ms", "us-central-1")
            # update_virtualservice_latency_k8s("checkoutservice-vs", "default", f"1ms", "us-south-1")
            # update_virtualservice_latency_k8s("checkoutservice-vs", "default", f"1ms", "us-east-1")
            # update_virtualservice_latency_k8s("checkoutservice-vs", "default", f"1ms", "us-west-1")
            
            output_dir = f"{sys_arg_dir_name}/{experiment.name}/{routing_rule}"
            print(f"mode: {mode}")
            print(f"routing_rule: {routing_rule}")
            set_env_vars_in_deployment(
                "slate-controller",
                "default",
                {
                    "JUMPING_INTERVAL_SECONDS": str(experiment.jumping_interval),
                    "JUMPING_WARM_START_THRESH": str(experiment.jumping_warm_start)
                }
            )
            utils.check_all_pods_are_ready()
            utils.create_dir(output_dir)
            utils.create_dir(f"{output_dir}/resource_alloc")
            utils.create_dir(f"{output_dir}/resource_usage")
            utils.create_dir(f"{output_dir}/cachewarmup")
            print(f"**** output_dir: {output_dir}")

            print("Warming caches...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_list = list()
                for workload in experiment.workloads:
                    wcopy = copy.deepcopy(workload)
                    wcopy.rps = [100]
                    wcopy.duration = [10]
                    future_list.append(executor.submit(utils.run_vegeta_go, wcopy, f"{output_dir}/cachewarmup", start_id=wcopy.start_id, num_ids=wcopy.num_ids))
                    time.sleep(0.1)
                for future in concurrent.futures.as_completed(future_list):
                    print(future.result())
            print("Caches warmed up.")

            for workload in experiment.workloads:
                CONFIG[f"RPS,{workload.cluster},{workload.req_type}"] = ",".join(map(str, workload.rps))
            for src, dst in experiment.normalization.items():
                for dst_key, dst_val in dst.items():
                    CONFIG[f"normalization,{src},{dst_key}"] = dst_val
            CONFIG["routing_rule"] = routing_rule
            CONFIG["capacity"] = 0
            CONFIG["hillclimb_interval"] = experiment.hillclimb_interval
            CONFIG['path'] = workload.path
            CONFIG['method'] = workload.method
            CONFIG["req_type"] = workload.req_type
            CONFIG["cluster"] = workload.cluster
            CONFIG["duration"] = workload.duration
            CONFIG["output_dir"] = output_dir
            CONFIG["load_bucket_size"] = 1
            utils.file_write_env_file(CONFIG)
            utils.file_write_config_file(CONFIG, f"{output_dir}/experiment-config.txt")
            # update env.txt and scp to slate-controller pod
            utils.kubectl_cp_from_host_to_slate_controller_pod("env.txt", "/app/env.txt")
            if mode == "runtime":
                utils.kubectl_cp_from_host_to_slate_controller_pod("poly-coef-cachethrash.csv", "/app/coef.csv")
                utils.kubectl_cp_from_host_to_slate_controller_pod("e2e-poly-coef-cachethrash.csv", "/app/e2e-coef.csv")
                slatelog = f"{benchmark_name}-trace.csv"
                utils.kubectl_cp_from_host_to_slate_controller_pod(slatelog, "/app/trace.csv")
                # t=5
                # print(f"sleep for {t} seconds to wait for the training to be done in global controller")
                # for i in range(t):
                #     time.sleep(1)
                #     print(f"start in {t-i} seconds")
            
            
            ############################################################################
            ############################################################################
            print(f"starting experiment at {datetime.now()}, expected to finish at {datetime.now() + timedelta(seconds=sum(workload.duration))}")
            
            if mode == "runtime":
                for (point, delay, targetregion) in experiment.injected_delay:
                    call_with_delay(point, update_virtualservice_latency_k8s, "recordservice-vs", "default", f"{delay}ms", targetregion)
                    print(f"injecting delay: {delay}ms at {point} seconds")
                start_node_cpu_monitoring(region_to_node, sum(workload.duration), f"{output_dir}/node_cpu_util.pdf")
                start_pod_cpu_monitoring(["sslateingress"], ["us-west-1", "us-central-1", "us-south-1", "us-east-1"], "default", sum(workload.duration), f"{output_dir}/frontend_pod_cpu_util.pdf")
                start_pod_cpu_monitoring(["record-service"], ["us-west-1", "us-central-1", "us-south-1", "us-east-1"], "default", sum(workload.duration), f"{output_dir}/record_pod_cpu_util.pdf")
            initial_byte_counts = utils.get_tc_byte_counts(network_interface, node_dict, inter_cluster_latency)
            print(f"initial_byte_counts: {initial_byte_counts}")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_list = list()
                for workload in experiment.workloads:
                    future_list.append(executor.submit(utils.run_vegeta_go, workload, output_dir, start_id=workload.start_id, num_ids=workload.num_ids))
                    # future_list.append(executor.submit(utils.run_vegeta, workload, output_dir))
                    time.sleep(0.1)
                for future in concurrent.futures.as_completed(future_list):
                    print(future.result())
            print("All clients have completed.")
            final_byte_counts = utils.get_tc_byte_counts(network_interface, node_dict, inter_cluster_latency)
            ############################################################################
            ############################################################################
            
            flist = ["/app/env.txt", "/app/endpoint_rps_history.csv", "/app/error.log", "/app/hillclimbing_distribution_history.csv", "/app/global_hillclimbing_distribution_history.csv"]
            for src_in_pod in flist:
                dst_in_host = f'{output_dir}/{src_in_pod.split("/")[-1]}'
                utils.kubectl_cp_from_slate_controller_to_host(src_in_pod, dst_in_host)
                # utils.run_command(f"python plot_rps.py {dst_in_host}")
            if mode == "profile":
                src_in_pod = "/app/trace_string.csv"
                dst_in_host = f"{output_dir}/trace.slatelog"
                utils.kubectl_cp_from_slate_controller_to_host(src_in_pod, dst_in_host)
            elif mode == "runtime":
                if "WATERFALL" in routing_rule or "SLATE" in routing_rule:
                    other_file_list = ["coefficient.csv", "routing_history.csv", "jumping_routing_history.csv", "jumping_latency.csv", "region_jumping_latency.csv", "expected_vs_actual_perf.csv"] # "constraint.csv", "variable.csv", "network_df.csv", "compute_df.csv"
                    for file in other_file_list:
                        src_in_pod = f"/app/{file}"
                        dst_in_host = f"{output_dir}/{file}"
                        utils.kubectl_cp_from_slate_controller_to_host(src_in_pod, dst_in_host)
            else:
                print(f"mode: {mode} is not supported")
                assert False
                
            # utils.run_command(f"python fast_plot.py --data_dir {output_dir}")
                
            # if routing_rule.startswith("SLATE-with-jumping") and os.path.exists(f"{output_dir}/SLATE-with-jumping-global-jumping_routing_history.csv"):
            if mode == "runtime":
                utils.run_command(f"python plot_gc_jumping.py {output_dir}/jumping_routing_history.csv {output_dir}/jumping_latency.csv {output_dir}/jumping_routing_rule_plots",required=False)
                utils.run_command(f"python plot_gc_jumping_paper.py {output_dir}/jumping_routing_history.csv {output_dir}/jumping_latency.csv {output_dir}/jumping_routing_rule_plots_paper",required=False)
                utils.run_command(f"python plot_region_latency.py {output_dir}/region_jumping_latency.csv {output_dir}/region_jumping_latency.pdf",required=False)
            utils.run_command(f"python ../plot_script/plot-vegeta-all.py {output_dir}", required=False)
            utils.run_command(f"python plot_endpoint_rps.py {output_dir}/endpoint_rps_history.csv {output_dir}/endpoint_rps.pdf",required=False)
            utils.run_command(f"python plot_endpoint_rps.py {output_dir}/endpoint_rps_history.csv {output_dir}/endpoint_rps_sslateingress.pdf sslateingress",required=False)
            utils.run_command(f"python plot_endpoint_rps.py {output_dir}/endpoint_rps_history.csv {output_dir}/endpoint_rps_corecontrast.pdf record-service",required=False)
            utils.run_command(f"python plot_expected_vs_actual.py {output_dir}/expected_vs_actual_perf.csv {output_dir}/expected_vs_actual_perf",required=False)
           # utils.run_command(f"python fast_plot_all.py --data_dir {output_dir}",required=False)
            # utils.run_command(f"python fast_plot.py --data_dir {output_dir}",required=False)
                # utils.run_command(f"python plot_gc_jumping ")
            costs = utils.calculate_egress_costs(initial_byte_counts, final_byte_counts, node_to_region)
            print(f"costs: {costs}")
            utils.print_and_save_egress_costs(costs, f"{output_dir}/egress_costs.json")

            '''end of one set of experiment'''
            
            ## restart slate-controller            
            ## add to cart restart
            # utils.restart_deploy(deploy=["slate-controller"], replicated_deploy=["sslateingress", "frontend", "productcatalogservice", "cartservice"], regions=["us-west-1"])
            
            ## checkout cart restart
            # utils.restart_deploy(deploy=["slate-controller"], replicated_deploy=['currencyservice', 'emailservice', 'cartservice', 'shippingservice', 'paymentservice', 'productcatalogservice','recommendationservice','frontend','sslateingress','checkoutservice'], regions=["us-west-1"])
            # utils.restart_deploy(replicated_deploy=["sslateingress", "record-service"], regions=["us-west-1"])
            utils.pkill_background_noise(node_dict)
            
            
        # if experiment_name.startswith("addtocart"):
        #     utils.restart_deploy(deploy=["slate-controller"], replicated_deploy=["sslateingress", "frontend", "productcatalogservice", "cartservice"], regions=["us-west-1"])
        # else:
            # utils.restart_deploy(deploy=["slate-controller"], replicated_deploy=['currencyservice', 'emailservice', 'cartservice', 'shippingservice', 'paymentservice', 'productcatalogservice','recommendationservice','frontend','sslateingress','checkoutservice'], regions=["us-west-1", "us-central-1"])  

            # savelogs(output_dir, services=['currencyservice', 'emailservice', 'cartservice', 'shippingservice', 'paymentservice', 'productcatalogservice','recommendationservice','frontend','sslateingress','checkoutservice'], regions=["us-central-1"])
            save_controller_logs(output_dir)
            utils.restart_deploy(deploy=["slate-controller"])
            
    # utils.run_command(f"python plot_script_from_main_branch/plot-vegeta-all.py {sys_arg_dir_name}", required=False)
    for node in node_dict:
        utils.run_command(f"ssh gangmuk@{node_dict[node]['hostname']} sudo tc qdisc del dev {network_interface} root", required=False, print_error=False)
        print(f"delete tc qdisc rule in {node_dict[node]['hostname']}")
            
if __name__ == "__main__":
    main()
