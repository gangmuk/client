import os
import subprocess
import json
from datetime import datetime
from kubernetes import client, config
import math
import time
import concurrent.futures
import sys
import os
import concurrent.futures
from datetime import datetime
import copy
import xml.etree.ElementTree as ET
from pprint import pprint
import atexit
import signal
import traceback
import yaml

def run_command_and_print(command, required=True, print_error=True, nonblock=False):
    """Run shell command and return its output, with real-time stdout streaming"""
    try:
        if nonblock:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            return True, "NotOutput-this-is-nonblocking-execution"
        else:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = []  # Collect output lines here

            # Stream stdout line by line
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()  # Ensure the output is printed immediately
                output.append(line)  # Store each line in the list

            # Wait for the process to complete
            process.wait()

            # Capture the complete output as a single string
            output_str = ''.join(output).strip()

            if process.returncode == 0:
                return True, output_str
            else:
                if print_error:
                    stderr_output = process.stderr.read()
                    print(f"ERROR command: {command}")
                    print(f"ERROR output: {stderr_output}")
                if required:
                    print("Exit...")
                    assert False
                else:
                    return False, stderr_output
    except subprocess.CalledProcessError as e:
        if print_error:
            print(f"ERROR command: {command}")
            print(f"ERROR output: {e.output.decode('utf-8').strip()}")
        if required:
            print("Exit...")
            assert False
        else:
            return False, e.output.decode('utf-8').strip()

import subprocess

def run_command(command, required=True, print_error=True, nonblock=False):
    """Run shell command and return its output or process handle.

    Args:
        command (str): The shell command to execute.
        required (bool): If True, the function will assert on failure.
        print_error (bool): If True, errors will be printed.
        nonblock (bool): If True, run the command non-blocking.

    Returns:
        tuple: 
            - If nonblock is False: (True, output) on success or (False, error) on failure.
            - If nonblock is True: (True, process) on success or (False, error) on failure.
    """
    try:
        if nonblock:
            # Start the process without waiting for it to complete
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return True, process
        else:
            # Run the command and wait for it to complete
            output = subprocess.check_output(
                command,
                shell=True,
                stderr=subprocess.STDOUT,
                text=True
            )
            return True, output.strip()
    except subprocess.CalledProcessError as e:
        if print_error:
            print(f"ERROR command: {command}")
            print(f"ERROR output: {e.output.strip()}")
        if required:
            print("Exiting due to required command failure...")
            raise  # Instead of assert False, it's better to raise an exception
        else:
            return False, e.output.strip()

def parse_xml(file_path):
    tree = ET.parse(file_path)
    return tree.getroot()


def get_nodename_and_ipaddr(file_path):
    node_dict = dict()
    namespaces = {
        'ns': 'http://www.geni.net/resources/rspec/3',
    }
    # Load and parse the XML file
    tree = ET.parse(file_path)
    root =  tree.getroot()
    # Extract node names and their IPv4 addresses
    nodes = root.findall('.//ns:node', namespaces)
    nodes_info = [
        (
            node.get('client_id'),
            # node.find('.//ns:host', namespaces).get('name'),
            [login.get('hostname') for login in node.findall('.//ns:services/ns:login', namespaces)][0],
            node.find('.//ns:host', namespaces).get('ipv4')
        )
        for node in nodes if node.find('.//ns:host', namespaces) is not None
    ]
    for node, hostname, ipaddr in nodes_info:
        node_dict[node] = {"hostname": hostname, "ipaddr": ipaddr}
    return node_dict



def get_pod_name_from_deploy(deployment_name, namespace='default'):
    config.load_kube_config()
    api_instance = client.AppsV1Api()
    try:
        success, pod_name = run_command(f"kubectl get pods -l app={deployment_name} -o custom-columns=:metadata.name")
        return pod_name
    except client.ApiException as e:
        print(f"Error fetching the pod name for deployment {deployment_name}: {e}")
        assert False
        
def check_file_exists_in_pod(pod_name, namespace, file_path):
    command = f"kubectl exec {pod_name} --namespace {namespace} -- sh -c '[ -f {file_path} ] && echo Exists || echo Does not exist'"
    success, ret = run_command(command, required=False)
    return success

def kubectl_cp_from_slate_controller_to_host(src_in_pod, dst_in_host, required=True):
    try:
        slate_controller_pod = get_pod_name_from_deploy("slate-controller")
        # if check_file_exists_in_pod(slate_controller_pod, "default", src_in_pod) == False:
        #     print(f"Skip scp. {src_in_pod} does not exist in the slate-controller pod")
        #     return
        # slate_controller_pod = run_command("kubectl get pods -l app=slate-controller -o custom-columns=:metadata.name")
        temp_file = "temp_file.txt"
        success, ret = run_command(f"kubectl cp {slate_controller_pod}:{src_in_pod} {temp_file}", required=required)
        if ret:
            # success
            run_command(f"mv {temp_file} {dst_in_host}", required=False)
            return True
        else:
            # fail
            print(f"\tSkip scp. {src_in_pod} does not exist in the slate-controller pod")
            return False
    except Exception as e:
        print(f"Error: {e}")
        print(f"- src_in_pod: {slate_controller_pod}:{src_in_pod}")
        print(f"- dst_in_host: {dst_in_host}")
        assert False

def kubectl_cp_from_host_to_slate_controller_pod(src_in_host, dst_in_pod):
    success, slate_controller_pod = run_command("kubectl get pods -l app=slate-controller -o custom-columns=:metadata.name")
    print(f"slate_controller_pod: {slate_controller_pod}")
    # if slate_controller_pod has more than one entry, get the start time of each pod and pick the newest one.
    if len(slate_controller_pod.split("\n")) > 1:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        slate_controller_pod = slate_controller_pod.split("\n")
        print(f"More than one slate_controller_pod ({slate_controller_pod}), picking the newest one...")
        newest = ""
        newest_time = -math.inf
        for pod in slate_controller_pod:
            try:
                pod_start_time = v1.read_namespaced_pod(pod, "default").status.start_time.timestamp()
                print(f"pod: {pod}, start_time: {pod_start_time}")
                if pod_start_time > newest_time:
                    newest = pod
                    newest_time = pod_start_time
            except client.exceptions.ApiException as e:
                # pod does not exist
                continue
        slate_controller_pod = newest
        print(f"Newest slate_controller_pod: {slate_controller_pod}")

    # print(f"Try kubectl_cp_from_host_to_slate_controller_pod")
    print(f"- src_in_host: {src_in_host}")
    print(f"- dst_in_pod: {slate_controller_pod}:{dst_in_pod}")
    tries = 0
    while tries < 3:
        success, ret = run_command(f"kubectl cp {src_in_host} {slate_controller_pod}:{dst_in_pod}", required=False)
        if success:
            return
        tries += 1
        print(f"Error: {ret}. Retrying...(try {tries})")
    print(f"Error: {ret}. Exiting...")
    assert False
    # print(f"finish scp from {src_in_host} to {slate_controller_pod}:{dst_in_pod}")
    


def convert_memory_to_mib(memory_usage):
    # Convert memory usage to MiB
    unit = memory_usage[-2:]  # Extract the unit (Ki, Mi, Gi)
    value = int(memory_usage[:-2])  # Extract the numeric value
    converted_value = 0
    if unit == "Ki":
        converted_value = value / 1024  # 1 MiB = 1024 KiB
    elif unit == "Mi":
        converted_value = value
    elif unit == "Gi":
        converted_value = value * 1024  # 1 GiB = 1024 MiB
    else:
        converted_value = value / (1024**2)  # Assume the value is in bytes if no unit
    return int(converted_value)

def convert_cpu_to_millicores(cpu_usage):
    converted_value = 0
    # Convert CPU usage to millicores
    if cpu_usage.endswith('n'):  # nanocores
        converted_value = int(cpu_usage.rstrip('n')) / 1000000
    elif cpu_usage.endswith('u'):  # assuming 'u' to be a unit close to millicores
        converted_value = int(cpu_usage.rstrip('u')) / 1000  # Convert assuming 'u' represents microcores
    else:
        converted_value = int(cpu_usage)  # Assuming direct millicore value
    return int(converted_value)

def get_pod_resource_usage(namespace='default'):
    config.load_kube_config()
    api_instance = client.CoreV1Api()
    custom_api = client.CustomObjectsApi()
    namespaces = ["default", "istio-system"]
    resource_allocation = "Namespace,Pod Name,Container Name,CPU Usage,Memory Usage\n"
    try:
        for namespace in namespaces:
            # Fetch current metrics for all pods in the namespace
            pod_metrics = custom_api.list_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=namespace,
                plural="pods"
            )
            metrics_map = {item["metadata"]["name"]: item for item in pod_metrics.get("items", [])}
            pods = api_instance.list_namespaced_pod(namespace, watch=False)
            for pod in pods.items:
                pod_name = pod.metadata.name
                for container in pod.spec.containers:
                    container_name = container.name
                    pod_metric = metrics_map.get(pod_name, {})
                    container_metric = next((c for c in pod_metric.get("containers", []) if c["name"] == container_name), None)
                    cpu_usage = "N/A"
                    memory_usage = "N/A"
                    if container_metric:
                        cpu_usage = container_metric.get("usage", {}).get("cpu", "N/A")
                        memory_usage = container_metric.get("usage", {}).get("memory", "N/A")
                        # print(f"before conversion, pod_name: {pod_name}, cpu_usage: {cpu_usage}, memory_usage: {memory_usage}")
                        if cpu_usage != "N/A":
                            cpu_usage = convert_cpu_to_millicores(cpu_usage)
                        if memory_usage != "N/A":
                            memory_usage = convert_memory_to_mib(memory_usage)
                        # print(f"after conversion, pod_name: {pod_name}, cpu_usage: {cpu_usage}, memory_usage: {memory_usage}")
                    resource_allocation += f"{namespace},{pod_name},{container_name},{cpu_usage},{memory_usage}\n"
    except Exception as e:
        print(f"Error: {e}")

    return resource_allocation

def get_pod_resource_allocation(namespace='default'):
    config.load_kube_config()
    api_instance = client.CoreV1Api()
    namespaces = ["default", "istio-system"]
    resource_allocation = "Namespace,Resource Type,Pod Name,Container Name,CPU Limit,CPU Request,Memory Limit,Memory Request\n"
    try:
        for namespace in namespaces:
            pods = api_instance.list_namespaced_pod(namespace, watch=False)
            for pod in pods.items:
                metadata = pod.metadata
                for container in pod.spec.containers:
                    # Fetch resource requests and limits
                    resources = container.resources
                    limits = resources.limits or {}
                    requests = resources.requests or {}

                    cpu_limit = limits.get("cpu", "N/A")
                    cpu_request = requests.get("cpu", "N/A")
                    mem_limit = limits.get("memory", "N/A")
                    mem_request = requests.get("memory", "N/A")

                    resource_allocation += f"{namespace},Pod,{metadata.name},{container.name},{cpu_limit},{cpu_request},{mem_limit},{mem_request}\n"
    except Exception as e:
        print(f"Error: {e}")
    return resource_allocation

# def record_pod_resource_allocation(wrk_log_path, target_cluster_rps):
#     if target_cluster_rps == 0:
#         return
#     with open(wrk_log_path, "a") as f:
#         f.write("\n-- start of resource allocation --\n")
#         resource_allocation = get_pod_resource_allocation()
#         f.write(resource_allocation)
#         f.write("-- end of resource allocation --\n")
        

# def record_pod_resource_usage(wrk_log_path, target_cluster_rps):
#     if target_cluster_rps == 0:
#         return
#     with open(wrk_log_path, "a") as f:
#         f.write("\n-- start of resource usage --\n")
#         resource_usage = get_pod_resource_usage()
#         f.write(resource_usage)
#         f.write("-- end of resource usage --\n\n")

def create_dir(directory):
    temp = 1
    original_dir = directory  # Preserve the original directory name
    while os.path.isdir(directory):
        directory = f"{original_dir}-{temp}"
        temp += 1
    os.makedirs(directory)
    print(f"Created directory: {directory}")
    return directory

def record_pod_resource_allocation(fn_prefix, resource_alloc_dir, target_cluster_rps):
    if target_cluster_rps == 0:
        return
    resource_alloc_file_path = f"{resource_alloc_dir}/{fn_prefix}-resource_alloc.csv"
    with open(resource_alloc_file_path, "w") as f:
        resource_allocation = get_pod_resource_allocation()
        f.write(resource_allocation)

def record_pod_resource_usage(fn_prefix, resource_usage_dir, target_cluster_rps):
    resource_usage_file_path = f"{resource_usage_dir}/{fn_prefix}-resource_usage.csv"
    if target_cluster_rps == 0:
        return
    with open(resource_usage_file_path, "w") as f:
        resource_usage = get_pod_resource_usage()
        f.write(resource_usage)
        
        
        

def sleep_and_print(sl):
    print(f"@@ sleep {sl} seconds")
    run_command(f"sleep {sl}")

def disable_istio():
    run_command("kubectl label namespace default istio-injection=disabled")
    print("@@ kubectl label namespace default istio-injection=disabled")
    sleep_and_print(10)

def enable_istio():
    run_command("kubectl label namespace default istio-injection=enabled")
    print("@@ kubectl label namespace default istio-injection=enabled")
    sleep_and_print(10)

def delete_wasm():
    run_command("kubectl delete -f /users/gangmuk/projects/slate-benchmark/wasmplugins.yaml")
    print("@@ kubectl delete -f /users/gangmuk/projects/slate-benchmark/wasmplugins.yaml")


def apply_wasm():
    run_command("kubectl apply -f /users/gangmuk/projects/slate-benchmark/wasmplugins.yaml")
    print("@@ kubectl apply -f /users/gangmuk/projects/slate-benchmark/wasmplugins.yaml")

def restart_wasm():
    delete_wasm()
    sleep_and_print(5)
    apply_wasm() # It looks like it is taking around 50s for wasm log to appear
    sleep_and_print(5)

def are_all_pods_ready(namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace)
    
    for pod in pods.items:
        # Skip if pod is terminating
        if pod.metadata.deletion_timestamp is not None:
            return False
            
        # Check all containers are ready against total containers
        if pod.status.container_statuses:
            ready_containers = sum(1 for container in pod.status.container_statuses if container.ready)
            total_containers = len(pod.status.container_statuses)
            
            # If not all containers are ready, the pod is not ready
            if ready_containers != total_containers:
                return False
        else:
            # If container statuses are not available, the pod is not ready
            return False
            
        # Check pod phase as a backup check
        if pod.status.phase != 'Running':
            return False
            
        # Check ready condition as another backup
        if pod.status.conditions:
            ready_condition = next((condition for condition in pod.status.conditions 
                                   if condition.type == 'Ready'), None)
            if not ready_condition or ready_condition.status != 'True':
                return False
                
    return True

def check_all_pods_are_ready(namespace='default'):
    ts1 = time.time()
    not_ready_pods = {}  # Dictionary to track not ready pods and their timestamps
    
    while True:
        ts2 = time.time()
        
        # Check if all pods are ready
        if are_all_pods_ready(namespace):
            break
            
        # Get list of not ready pods and their deployments
        current_not_ready = get_not_ready_pods_with_deployments(namespace)
        current_time = time.time()
        
        # Update tracking dictionary with new not ready pods
        for pod_name, deployment in current_not_ready.items():
            if pod_name not in not_ready_pods:
                not_ready_pods[pod_name] = {
                    'deployment': deployment,
                    'first_seen': current_time
                }
        
        # Check for pods that have been not ready for more than 30 seconds
        pods_to_restart = {}
        for pod_name, data in list(not_ready_pods.items()):
            # If pod is now ready, remove from tracking
            if pod_name not in current_not_ready:
                not_ready_pods.pop(pod_name)
                continue
                
            # If pod has been not ready for more than 30 seconds
            if current_time - data['first_seen'] > 30:
                deployment = data['deployment']
                if deployment not in pods_to_restart:
                    pods_to_restart[deployment] = []
                pods_to_restart[deployment].append(pod_name)
                
        # Restart deployments for pods that have been not ready for too long
        for deployment, pods in pods_to_restart.items():
            print(f"Pods {', '.join(pods)} not ready for more than 30 seconds. Restarting deployment {deployment}")
            subprocess.run(["kubectl", "rollout", "restart", "deploy", deployment, "-n", namespace])
            
            # Remove restarted pods from tracking
            for pod in pods:
                not_ready_pods.pop(pod)
        
        print(f"Waiting for all pods to be ready, {int(time.time()-ts1)} seconds has passed: {not_ready_pods}")
        
        # Sleep logic
        sl = time.time() - ts2
        if sl <= 0:
            continue
        time.sleep(sl)
        
    print("All pods are ready")

# Helper function to get not ready pods and their deployment names
def get_not_ready_pods_with_deployments(namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace)
    
    not_ready_pods = {}
    
    for pod in pods.items:
        # Check if pod is not ready
        is_ready = True
        
        # Check if pod is terminating
        if pod.metadata.deletion_timestamp is not None:
            is_ready = False
        
        # Check if all containers are ready
        elif pod.status.container_statuses:
            ready_containers = sum(1 for container in pod.status.container_statuses if container.ready)
            total_containers = len(pod.status.container_statuses)
            
            if ready_containers != total_containers:
                is_ready = False
        else:
            is_ready = False
            
        # If pod is not ready, add it to our dictionary
        if not is_ready:
            pod_name = pod.metadata.name
            deployment_name = get_deployment_for_pod(pod, namespace)
            if deployment_name:
                not_ready_pods[pod_name] = deployment_name
                
    return not_ready_pods

# Helper function to get deployment name for a pod
def get_deployment_for_pod(pod, namespace='default'):
    # Extract deployment name from owner references
    if pod.metadata.owner_references:
        for owner_ref in pod.metadata.owner_references:
            if owner_ref.kind == "ReplicaSet":
                # Get the ReplicaSet to find its owner Deployment
                rs_name = owner_ref.name
                config.load_kube_config()
                apps_v1 = client.AppsV1Api()
                
                try:
                    rs = apps_v1.read_namespaced_replica_set(name=rs_name, namespace=namespace)
                    
                    # Find the Deployment owner of this ReplicaSet
                    if rs.metadata.owner_references:
                        for rs_owner in rs.metadata.owner_references:
                            if rs_owner.kind == "Deployment":
                                return rs_owner.name
                except client.exceptions.ApiException:
                    pass
    
    # Alternatively, try to get deployment from pod labels (common convention)
    if pod.metadata.labels and 'app' in pod.metadata.labels:
        return pod.metadata.labels['app']
        
    return None

def restart_deploy(deploy=[], replicated_deploy=[], exclude=[], regions=[]):
    ts = time.time()
    config.load_kube_config()
    api_instance = client.AppsV1Api()
    for d in replicated_deploy:
        for r in regions:
            deploy.append(d + "-" + r)
    try:
        deployments = api_instance.list_namespaced_deployment(namespace="default")
        for deployment in deployments.items:
            if deployment.metadata.name not in exclude and deployment.metadata.name in deploy:
                run_command(f"kubectl rollout restart deploy {deployment.metadata.name} ")
    except client.ApiException as e:
        print("Exception when calling AppsV1Api->list_namespaced_deployment: %s\n" % e)

def add_latency_rules(src_host, interface, dst_node_ip, delay):
    if delay == 0:
        print(f"skip add_latency_rules, delay: {delay}")
        return
    class_id = f"1:{delay}"
    handle_id = delay
    run_command(f'ssh gangmuk@{src_host} sudo tc class add dev {interface} parent 1: classid {class_id} htb rate 100mbit', required=False, print_error=False)
    run_command(f'ssh gangmuk@{src_host} sudo tc qdisc add dev {interface} parent {class_id} handle {handle_id}: netem delay {delay}ms', required=False, print_error=False)
    run_command(f'ssh gangmuk@{src_host} sudo tc filter add dev {interface} protocol ip parent 1:0 prio 1 u32 match ip dst {dst_node_ip} flowid {class_id}')


def start_background_noise(node_dict, cpu_noise=0, victimize_node="", victimize_cpu=0):
    for node in node_dict:
        if node == "node0":
            print("[NOTE]SKIP start_background_noise in node0. node0 is control plane node")
            continue
        
        if node == "node5":
            print("[NOTE]SKIP start_background_noise in node5. node5 is running slate-controller")
            continue
        if node == "node6":
            print("[NOTE]SKIP start_background_noise in node6. node6 is running igw")
            continue
        
        nodenoise = cpu_noise
        if node == victimize_node:
            nodenoise = victimize_cpu
        # print(f"Try to run background-noise -cpu={cpu_noise} in {node_dict[node]['hostname']}")
        print("ADITYA: ", f"ssh gangmuk@{node_dict[node]['hostname']} 'nohup /users/gangmuk/projects/slate-benchmark/background-noise/background-noise -cpu={nodenoise} &'")
        run_command(f"ssh gangmuk@{node_dict[node]['hostname']} 'nohup /users/gangmuk/projects/slate-benchmark/background-noise/background-noise -cpu={nodenoise} > /dev/null 2>&1 &'", nonblock=False)
        # run_command(f"ssh gangmuk@{node_dict[node]['hostname']} '", nonblock=False)

        print(f"background-noise -cpu={cpu_noise} in {node_dict[node]['hostname']}")
        
        
def recalculate_capacity(capacity, rps_dict):
    total_capacity = capacity * len(rps_dict)
    total_demand = sum(rps_dict.values())
    if total_demand > total_capacity:
        return total_demand // len(rps_dict)
    else:
        return capacity
    
def pkill_background_noise(node_dict):
    for node in node_dict:
        pkill_command = 'pkill -f background-noise'
        run_command(f"ssh gangmuk@{node_dict[node]['hostname']} {pkill_command}", required=False, print_error=False)
        print(f"{pkill_command} in {node_dict[node]['hostname']}")
        
def apply_all_tc_rule(interface, inter_cluster_latency, node_dict):
        for src_node in inter_cluster_latency:
            src_host = node_dict[src_node]['hostname']
            run_command(f'ssh gangmuk@{src_host} sudo tc qdisc add dev {interface} root handle 1: htb')
            for dst_node in inter_cluster_latency[src_node]:
                if src_node == dst_node:
                    continue
                dst_node_ip = node_dict[dst_node]['ipaddr'] 
                delay = inter_cluster_latency[src_node][dst_node]
                add_latency_rules(src_host, interface, dst_node_ip, delay)
                print(f"Added {delay}ms from {src_host}({src_node}) to {dst_node_ip}({dst_node})")
                
def delete_tc_rule_in_client(nic, node_dict):
    for node in node_dict:
        run_command(f"ssh gangmuk@{node_dict[node]['hostname']} sudo tc qdisc del dev {nic} root", required=False, print_error=False)
        print(f"delete tc qdisc rule in {node_dict[node]['hostname']}")

def file_write_experiment_config(config_dict, config_file_path):
    with open(config_file_path, "w") as f:
        info = "-- start of config --\n"
        for key, value in config_dict.items():
            info += f"{key},{value}\n"
        info += "-- end of config --\n\n"
        f.write(info)
    print(f"config file write to {config_file_path}")


# One experiment means a set of workloads.
# If you want to run addtocart, 100RPS, 60s, to west cluster, and addtocart, 200RPS, 60s, to east cluster, you need to create two workloads.
# Only one workload is allowed for one cluster.
class Experiment:
    def __init__(self):
        self.name = None
        self.workloads = list()
        self.workload_names = set()
        self.hillclimb_interval = 0
        self.injected_delay = []
        self.delay_injection_point =0 
        self.limit_val = ""
        self.hash_mod = 1
        self.normalization = dict()
        self.workload_raw = dict()
        
    def set_name(self, name):
        self.name = name
    
    def set_hillclimb_interval(self, interval):
        self.hillclimb_interval = interval
    
    def set_injected_delay(self, delay):
        self.injected_delay = delay
    
    def set_delay_injection_point(self, delay_injection_point):
        self.delay_injection_point = delay_injection_point

    def set_limit_val(self, limit_val):
        self.limit_val = limit_val
    
    def add_workload(self, workload):
        if workload.name in self.workload_names:
            print(f"ERROR: {workload.name} workload already exists in this experiment")
            print("exit...")
            exit()
        self.workload_names.add(workload.name)
        self.workloads.append(workload)
        print(f"Added workload: {workload.cluster}, {workload.req_type}, {workload.rps}, {workload.duration}")
        
    def print_experiment(self):
        print(f"Experiment Name: {self.name}")
        for workload in self.workloads:
            workload.print_workload()

def file_write_env_file(CONFIG):
    with open("env.txt", "w") as file:
        for key, value in CONFIG.items():
            file.write(f"{key},{value}\n")

def file_write_config_file(CONFIG, config_file_path):
    with open(config_file_path, "w") as file:
        file.write("-- start of config --\n")
        for key, value in CONFIG.items():
            file.write(f"{key},{value}\n")
        file.write("-- end of config --")


def update_wasm_env_var(namespace, plugin_name, hillclimbing_key, hillclimbing_value):
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
    print(f"WASM key {hillclimbing_key} value updated to {hillclimbing_value} in WasmPlugin '{plugin_name}'.")
                   
# One workload means one client. One client means one request type and a list of RPS & Duration>.
class Workload:
    def __init__(self, cluster: str, req_type: str, rps: list, duration: list, method: str, path: str, hdrs: dict = {}, endpoint=""):
        self.cluster = cluster
        self.req_type = req_type
        self.rps = rps
        self.duration = duration
        self.method = method
        self.path = path
        self.hdrs = hdrs
        self.name = f"{cluster}-{req_type}"
        # self.name = f"{cluster}-{req_type}-{rps}"
        self.endpoint = endpoint
        if len(rps) != len(duration):
            print(f"ERROR: rps and duration length mismatch")
            print(f"rps: {rps}")
            print(f"duration: {duration}")
            print("exit...")
            assert False
    
    def print_workload(self):
        print(f"cluster: {self.cluster}")
        print(f"req_type: {self.req_type}")
        print(f"rps: {self.rps}")
        print(f"duration: {self.duration}s")
        
import yaml

def write_client_yaml_with_config(default_yaml_file: str, yaml_file: str, workload, output_dir: str):
    with open(default_yaml_file, 'r') as file:
        yaml_data = yaml.safe_load(file)
    try:
        yaml_data['clients'][0]['cluster'] = workload.cluster
        yaml_data['clients'][0]['headers']['x-slate-destination'] = workload.cluster
        for key, value in workload.hdrs.items():
            yaml_data['clients'][0]['headers'][key] = value
        
        # Ensure 'workload' list is long enough, append entries if needed
        while len(yaml_data['clients'][0]['workload']) < len(workload.rps):
            yaml_data['clients'][0]['workload'].append({'duration': None, 'rps': None})

        for i in range(len(workload.rps)):
            yaml_data['clients'][0]['workload'][i]['duration'] = f"{workload.duration[i]}s"
            yaml_data['clients'][0]['workload'][i]['rps'] = workload.rps[i]

        yaml_data['clients'][0]['method'] = workload.method
        yaml_data['clients'][0]['path'] = workload.path
        yaml_data['clients'][0]['rq_timeout'] = '2s'
        yaml_data['stats_output_folder'] = output_dir
    except Exception as e:
        print(f"ERROR: {e}")
        print("exit...")
        assert False
    
    # Overwrite the new file if it exists
    with open(yaml_file, 'w') as file:
        yaml.dump(yaml_data, file)

    print(f"Updated YAML data has been written to {yaml_file}, overwriting if it already existed.")

            
def run_newer_generation_client(workload, output_dir):
    print(f"start {workload.req_type} RPS {workload.rps} to {workload.cluster} cluster for {workload.duration}s")
    client_yaml_file = f'config-{workload.name}.yaml'
    write_client_yaml_with_config("config.yaml", client_yaml_file, workload, output_dir)
    run_command(f'cp {client_yaml_file} {output_dir}/{client_yaml_file}')
    run_command(f"./client --config={client_yaml_file}")


# def run_vegeta(workload, output_dir):
#     for i in range(len(workload.rps)):
#         print(f"start {workload.req_type} RPS {workload.rps[i]} to {workload.cluster} cluster for {workload.duration[i]}s")
        
#         cmd_1 = f"echo '{workload.method} {workload.endpoint}{workload.path}' | /users/gangmuk/projects/client/vegeta attack -rate={workload.rps[i]} -duration={workload.duration[i]}s -timeout=5s"
        
#         headers = ""
#         for key, value in workload.hdrs.items():
#             headers += f" -header='{key}: {value}'"
        
#         cmd_2 = f" -header='x-slate-destination: {workload.cluster}' | /users/gangmuk/projects/client/vegeta report -type='hdrplot' > {output_dir}/{workload.name}-{workload.rps[i]}-{workload.duration[i]}.output"
        
#         cmd = cmd_1 + headers + cmd_2
#         run_command(cmd)


def run_vegeta(workload, output_dir):
    for i in range(len(workload.rps)):
        print(f"start-{i}, {workload.req_type} RPS {workload.rps[i]} to {workload.cluster} cluster for {workload.duration[i]}s")
        
        # cmd_1 = f"echo '{workload.method} {workload.endpoint}{workload.path}' | /users/gangmuk/projects/client/vegeta attack -rate={workload.rps[i]} -duration={workload.duration[i]}s -timeout=5s"
        cmd_1 = f"echo '{workload.method} {workload.endpoint}{workload.path}' | /users/gangmuk/projects/client/vegeta attack -rate={workload.rps[i]} -duration={workload.duration[i]}s"
        
        headers = ""
        for key, value in workload.hdrs.items():
            headers += f" -header='{key}: {value}'"
        
        cmd_2 = f" -header='x-slate-destination: {workload.cluster}'"
        cmd_3 = f"| tee {output_dir}/{i}-{workload.rps[i]}RPS-{workload.duration[i]}s.{workload.req_type}.{workload.cluster}.results.bin | /users/gangmuk/projects/client/vegeta report > {output_dir}/{i}-{workload.rps[i]}RPS-{workload.duration[i]}s.{workload.req_type}.{workload.cluster}.stats.txt"
        
        cmd = cmd_1 + headers + cmd_2 + cmd_3
        
        print(f"vegeta cmd: {cmd}")
        # echo 'POST http://node5.slate-gm.istio-pg0.cloudlab.umass.edu:32048/cart/checkout?email=fo%40bar.com&street_address=405&zip_code=945&city=Fremont&state=CA&country=USA&credit_card_number=5555555555554444&credit_card_expiration_month=12&credit_card_expiration_year=2025&credit_card_cvv=222' | /users/gangmuk/projects/client/vegeta attack -rate=2000 -duration=10s -timeout=5s -header='x-slate-destination: west' | tee vegeta-test.results.bin | /users/gangmuk/projects/client/vegeta report > vegeta_stats.txt
        
        # print(f"vegeta cmd: {cmd}")
        run_command(cmd)
        
        # echo 'POST http://node5.slate-gm.istio-pg0.cloudlab.umass.edu:32048/cart/checkout?email=fo%40bar.com&street_address=405&zip_code=945&city=Fremont&state=CA&country=USA&credit_card_number=5555555555554444&credit_card_expiration_month=12&credit_card_expiration_year=2025&credit_card_cvv=222' | /users/gangmuk/projects/client/vegeta attack -rate=100 -duration=10s -timeout=5s -header='x-slate-destination: south' | tee results.bin | /users/gangmuk/projects/client/vegeta report


def extract_latencies(json_file, output_dir, rps, duration):
    with open(json_file, "r") as f:
        records = []
        for line in f:
            data = json.loads(line)
            latency = int(data["latency"]) / 1_000_000  # Convert latency to milliseconds
            timestamp = data["timestamp"]  # ISO 8601 format
            status_code = data["code"]  # HTTP status code
            records.append((timestamp, latency, status_code))
    output_csv = f"{output_dir}/latencies-{rps}-{duration}.csv"
    with open(output_csv, "w") as f:
        f.write("Timestamp,Latency(ms),StatusCode\n")
        for record in records:
            f.write(f"{record[0]},{record[1]:.3f},{record[2]}\n")
    print(f"Saved detailed request metrics to {output_csv}")


def run_stress(c, vm, vm_bytes, start_in_seconds, duration, node_dict, target_node):
    time.sleep(start_in_seconds)
    # run_command(f"ssh gangmuk@{node_dict[node]['hostname']} 'nohup /users/gangmuk/projects/slate-benchmark/background-noise/background-noise -cpu={nodenoise} > /dev/null 2>&1 &'", nonblock=False)
    full_cmd = f"ssh gangmuk@{node_dict[target_node]['hostname']} 'nohup stress -c {c} --vm {vm} --vm-bytes {vm_bytes}M --timeout {duration} > /dev/null 2>&1 &'"
    print(f"run_stress cmd: {full_cmd}")
    run_command(full_cmd, nonblock=False)
    
    
def pkill_stress(node_dict):
    for node in node_dict:
        pkill_command = 'pkill -f stress'
        run_command(f"ssh gangmuk@{node_dict[node]['hostname']} {pkill_command}", required=False, print_error=False)
        print(f"{pkill_command} in {node_dict[node]['hostname']}")
