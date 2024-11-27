from kubernetes import client, config
import yaml
import os, subprocess
from collections import defaultdict
import json

replMap = {
    "gcr.io/google-samples/microservices-demo/frontend:v0.10.1": "docker.io/adiprerepa/boutique-frontend:latest",
    "gcr.io/google-samples/microservices-demo/checkoutservice:v0.10.1": "docker.io/adiprerepa/boutique-checkout:latest",
    "gcr.io/google-samples/microservices-demo/recommendationservice:v0.10.1": "docker.io/adiprerepa/boutique-recommendation:latest",
}
def scale_deployments(namespace, replica_count, exclude_deployments=[]):
    # Load kubeconfig (assumes you have access to the cluster)
    config.load_kube_config()

    # Create an instance of the API class
    apps_v1 = client.AppsV1Api()

    # Get all deployments in the specified namespace
    deployments = apps_v1.list_namespaced_deployment(namespace)

    for deployment in deployments.items:
        # Skip deployments in the exclude list
        if deployment.metadata.name in exclude_deployments:
            continue
        
        # Define the new number of replicas
        body = {
            "spec": {
                "replicas": replica_count
            }
        }

        # Scale the deployment
        response = apps_v1.patch_namespaced_deployment_scale(
            name=deployment.metadata.name,
            namespace=namespace,
            body=body
        )

        print(f"Scaled deployment {deployment.metadata.name} to {replica_count} replicas")

def update_deployment_images():
    # Create the AppsV1 API client
    config.load_kube_config()
    apps_v1 = client.AppsV1Api()

    # Get all deployments in all namespaces
    deployments = apps_v1.list_deployment_for_all_namespaces().items

    for deployment in deployments:
        namespace = deployment.metadata.namespace
        name = deployment.metadata.name
        containers = deployment.spec.template.spec.containers
        
        # Flag to determine if an update is needed
        update_needed = False

        # Go through all the containers in the deployment
        for container in containers:
            current_image = container.image
            if current_image in replMap:
                new_image = replMap[current_image]
                print(f"Updating image in deployment {name} in namespace {namespace}: {current_image} -> {new_image}")
                container.image = new_image
                update_needed = True

        # If any container image was updated, patch the deployment
        if update_needed:
            # Patch the deployment with the updated image
            try:
                apps_v1.patch_namespaced_deployment(
                    name=name,
                    namespace=namespace,
                    body=deployment
                )
                print(f"Successfully updated deployment {name} in namespace {namespace}")
            except Exception as e:
                print(f"Error updating deployment {name} in namespace {namespace}: {e}")

def change_load_balancing_policy_to_round_robin():
    # Load kubeconfig (default location is ~/.kube/config)
    config.load_kube_config()

    # Create an instance of the API class for custom resources
    custom_api = client.CustomObjectsApi()

    # Fetch all DestinationRules across all namespaces
    group = 'networking.istio.io'
    version = 'v1beta1'
    plural = 'destinationrules'

    # Fetch DestinationRules across all namespaces
    destinationrules = custom_api.list_cluster_custom_object(group=group, version=version, plural=plural)

    for rule in destinationrules['items']:
        namespace = rule['metadata']['namespace']
        name = rule['metadata']['name']

        # Ensure 'trafficPolicy' exists in the spec, if not, create it
        if 'trafficPolicy' not in rule['spec']:
            rule['spec']['trafficPolicy'] = {}

        # Ensure 'loadBalancer' exists under 'trafficPolicy', if not, create it
        if 'loadBalancer' not in rule['spec']['trafficPolicy']:
            rule['spec']['trafficPolicy']['loadBalancer'] = {}

        # Update load balancing policy to round robin
        rule['spec']['trafficPolicy']['loadBalancer'] = {
            'simple': 'ROUND_ROBIN'
        }

        # Update the DestinationRule with the modified spec
        custom_api.patch_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
            name=name,
            body=rule
        )
        print(f"Updated DestinationRule '{name}' in namespace '{namespace}' to use ROUND_ROBIN load balancing.")

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
# scale_deployments("default", 4, exclude_deployments=["slate-controller"])
def update_image_pull_policy():
    # Load kube config
    config.load_kube_config()

    # Create an instance of the API class for interacting with deployments
    api_instance = client.AppsV1Api()

    # List all deployments in the default namespace
    namespace = 'default'
    deployments = api_instance.list_namespaced_deployment(namespace=namespace)

    for deployment in deployments.items:
        # Skip the slate-controller deployment
        if deployment.metadata.name == 'slate-controller':
            continue

        # Iterate over containers and update the imagePullPolicy
        updated = False
        for container in deployment.spec.template.spec.containers:
            if container.image_pull_policy != 'IfNotPresent':
                container.image_pull_policy = 'IfNotPresent'
                updated = True
        
        # Patch the deployment only if it was updated
        if updated:
            # Create a patch body
            body = {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": container.name,
                                    "imagePullPolicy": container.image_pull_policy
                                } for container in deployment.spec.template.spec.containers
                            ]
                        }
                    }
                }
            }

            # Apply the patch to the deployment
            api_instance.patch_namespaced_deployment(
                name=deployment.metadata.name,
                namespace=namespace,
                body=body
            )
            print(f"Updated imagePullPolicy to IfNotPresent for deployment: {deployment.metadata.name}")


def set_retries_to_zero(namespace="default"):
    # Load Kubernetes configuration (assuming you have access via kubectl)
    config.load_kube_config()

    # Create an API client for Custom Objects and Networking API
    api_instance = client.CustomObjectsApi()
    networking_v1beta1 = client.NetworkingV1beta1Api()

    # List all DestinationRule resources in the specified namespace
    destination_rules = api_instance.list_namespaced_custom_object(
        group="networking.istio.io",
        version="v1beta1",
        namespace=namespace,
        plural="destinationrules"
    )

    for rule in destination_rules.get('items', []):
        spec = rule.get('spec', {})
        traffic_policy = spec.get('trafficPolicy', {})
        
        # Set retries attempts to 0 if 'retries' exists in the traffic policy
        if 'retries' in traffic_policy:
            traffic_policy['retries']['attempts'] = 0
        else:
            # Add retry policy if it doesn't exist
            traffic_policy['retries'] = {
                'attempts': 0,
                'perTryTimeout': '2s'  # Optionally set a timeout, adjust as necessary
            }

        # Update the destination rule with the modified spec
        rule_name = rule['metadata']['name']
        print(f"Updating {rule_name}...")

        # Apply the modified trafficPolicy back to the DestinationRule
        rule['spec']['trafficPolicy'] = traffic_policy
        api_instance.replace_namespaced_custom_object(
            group="networking.istio.io",
            version="v1beta1",
            namespace=namespace,
            plural="destinationrules",
            name=rule_name,
            body=rule
        )

    print("Updated all DestinationRules in namespace:", namespace)
# update_image_pull_policy()
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
        print(f"Error fetching pod list: {e}")

import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def compute_traffic_matrix(df, src_service, dst_service):
    """
    Computes the traffic matrix for the specified source and destination services.

    Args:
        df (pd.DataFrame): Input DataFrame containing traffic data.
        src_service (str): Source service name to filter.
        dst_service (str): Destination service name to filter.

    Returns:
        pd.DataFrame: Pivot table representing the traffic matrix.
    """
    # Ensure necessary columns are present
    required_columns = {'src_svc', 'dst_svc', 'dst_cid', 'dst_endpoint', 'weight', 'src_cid'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}, but has columns: {df.columns}")

    # Filter the DataFrame for the specified source and destination services
    filtered = df[
        (df['src_svc'] == src_service) & 
        (df['dst_svc'] == dst_service)
    ]

    # Create a MultiIndex for the columns using dst_cid and dst_endpoint
    traffic_matrix = filtered.pivot_table(
        index=['src_cid', 'src_endpoint'],    
        columns=['dst_cid', 'dst_endpoint'],
        values='weight',
        aggfunc='sum',
        fill_value=0
    )

    return traffic_matrix

def jump_towards_optimizer_desired(starting_df: pd.DataFrame, desired_df: pd.DataFrame, 
                                   src_service: str, dst_service: str, 
                                   cur_convex_comb_value: float) -> pd.DataFrame:
    """
    Computes the convex combination of two traffic matrices based on the given combination value.

    Args:
        starting_df (pd.DataFrame): The starting traffic matrix DataFrame.
        desired_df (pd.DataFrame): The desired traffic matrix DataFrame.
        src_service (str): Source service name to filter.
        dst_service (str): Destination service name to filter.
        cur_convex_comb_value (float): The convex combination factor (between 0 and 1).

    Returns:
        pd.DataFrame: The new traffic matrix as a DataFrame in the same format as the inputs.
    """
    # Validate cur_convex_comb_value
    if not (0 <= cur_convex_comb_value <= 1):
        raise ValueError("cur_convex_comb_value must be between 0 and 1.")

    required_columns = {'src_svc', 'dst_svc'}
    for df_name, df in zip(['starting_df', 'desired_df'], [starting_df, desired_df]):
        if not required_columns.issubset(df.columns):
            raise ValueError(f"{df_name} must contain columns: {required_columns} (has columns: {df.columns})")

    # Compute traffic matrices for starting and desired DataFrames
    starting_matrix = compute_traffic_matrix(starting_df, src_service, dst_service)
    desired_matrix = compute_traffic_matrix(desired_df, src_service, dst_service)
    
    # Identify all unique (src_cid, src_endpoint) and (dst_cid, dst_endpoint) across both matrices
    all_src = starting_matrix.index.union(desired_matrix.index)
    all_dst = starting_matrix.columns.union(desired_matrix.columns)
    
    # Reindex both matrices to include all sources and destinations
    starting_matrix = starting_matrix.reindex(index=all_src, columns=all_dst, fill_value=0)
    desired_matrix = desired_matrix.reindex(index=all_src, columns=all_dst, fill_value=0)
    
    # Compute the convex combination
    combined_matrix = (1 - cur_convex_comb_value) * starting_matrix + cur_convex_comb_value * desired_matrix
    combined_matrix = combined_matrix.round(6)
    logger.info(f"Combined traffic matrix:\n{combined_matrix}\nStarting matrix:\n{starting_matrix}\nDesired matrix:\n{desired_matrix}")
    
    # Transform the combined matrix back into a DataFrame
    combined_df = combined_matrix.reset_index().melt(id_vars=['src_cid', 'src_endpoint'], 
                                                   var_name=['dst_cid', 'dst_endpoint'], 
                                                   value_name='weight')
    combined_df = combined_df[combined_df['weight'] > 0].reset_index(drop=True)
    
    # Merge with starting_df and desired_df to get 'total' and 'flow' information
    # **Important Correction**: Include 'src_endpoint' and 'dst_endpoint' in the merge keys
    starting_totals = starting_df[
        (starting_df['src_svc'] == src_service) & 
        (starting_df['dst_svc'] == dst_service)
    ][['src_cid', 'src_endpoint', 'dst_cid', 'dst_endpoint', 'total']].drop_duplicates()
    
    desired_totals = desired_df[
        (desired_df['src_svc'] == src_service) & 
        (desired_df['dst_svc'] == dst_service)
    ][['src_cid', 'src_endpoint', 'dst_cid', 'dst_endpoint', 'total']].drop_duplicates()
    
    # Merge combined_df with starting_totals
    combined_df = combined_df.merge(
        starting_totals,
        on=['src_cid', 'src_endpoint', 'dst_cid', 'dst_endpoint'],
        how='left',
        suffixes=('', '_start')
    )
    
    # Merge combined_df with desired_totals
    combined_df = combined_df.merge(
        desired_totals,
        on=['src_cid', 'src_endpoint', 'dst_cid', 'dst_endpoint'],
        how='left',
        suffixes=('', '_desired')
    )
    
    # Fill 'total' from starting_df; if missing, use desired_df's 'total'; else set to 1 to avoid division by zero
    combined_df['total'] = combined_df['total'].fillna(combined_df['total_desired']).fillna(1)
    
    # Compute 'flow' as weight * total
    combined_df['flow'] = combined_df['weight'] * combined_df['total']
    
    # Add 'src_svc' and 'dst_svc' columns
    combined_df['src_svc'] = src_service
    combined_df['dst_svc'] = dst_service
    
    # Optional: Enforce weight limits if necessary
    # Here, we clip weights to ensure they stay within [0, 1]
    combined_df['weight'] = combined_df['weight'].clip(lower=0, upper=1.0)
    
    # Reorder and select columns to match the original format
    final_df = combined_df[
        ['src_svc', 'dst_svc', 'src_endpoint', 'dst_endpoint',
         'src_cid', 'dst_cid', 'flow', 'total', 'weight']
    ]
    
    # Sort and reset index
    final_df = final_df.sort_values(by=['src_cid', 'dst_cid']).reset_index(drop=True)
    
    return final_df

def process_workloads_to_stages(workloads, total_duration):
    """
    Transforms the workloads dict into a list of stages.
    Each stage is a tuple:
    (stage_duration, west_singlecore_rps, west_multicore_rps,
     east_singlecore_rps, east_multicore_rps,
     central_singlecore_rps, central_multicore_rps,
     south_singlecore_rps, south_multicore_rps)
    
    Parameters:
    - workloads: Dict containing per-region, per-request-type RPS configurations.
    - total_duration: Total duration of the experiment in seconds.
    
    Returns:
    - final_stages: List of 9-tuples representing each stage.
    """
    # Collect all unique start times
    unique_times = set()
    for region, req_types in workloads.items():
        for req_type, timings in req_types.items():
            for timing in timings:
                unique_times.add(timing[0])

    sorted_times = sorted(unique_times)

    # Ensure that the first start time is 0
    if sorted_times[0] != 0:
        sorted_times = [0] + sorted_times

    # Initialize current RPS for all regions and request types
    regions = ["west", "east", "central", "south"]
    request_types = ["singlecore", "multicore"]
    current_rps = {region: {req_type: 0 for req_type in request_types} for region in regions}

    stages = []

    # Iterate through each start time and update RPS accordingly
    for time in sorted_times:
        for region, req_types in workloads.items():
            for req_type, timings in req_types.items():
                for timing in timings:
                    if timing[0] == time:
                        current_rps[region][req_type] = timing[1]

        # Create a snapshot of current RPS
        snapshot = {
            "west_singlecore": current_rps["west"]["singlecore"],
            "west_multicore": current_rps["west"]["multicore"],
            "east_singlecore": current_rps["east"]["singlecore"],
            "east_multicore": current_rps["east"]["multicore"],
            "central_singlecore": current_rps["central"]["singlecore"],
            "central_multicore": current_rps["central"]["multicore"],
            "south_singlecore": current_rps["south"]["singlecore"],
            "south_multicore": current_rps["south"]["multicore"],
        }

        stages.append((time, snapshot))

    # Sort stages by start_time
    stages = sorted(stages, key=lambda x: x[0])

    # Merge stages with the same configuration
    merged_stages = []
    previous_snapshot = None
    for stage in stages:
        if previous_snapshot and stage[1] == previous_snapshot:
            continue  # Skip if the configuration hasn't changed
        merged_stages.append(stage)
        previous_snapshot = stage[1]

    # Convert to list of 9-tuples with duration
    final_stages = []
    for i in range(len(merged_stages)):
        stage_time = merged_stages[i][0]
        # Determine the duration of the stage
        if i + 1 < len(merged_stages):
            next_stage_time = merged_stages[i + 1][0]
            duration = next_stage_time - stage_time
        else:
            duration = total_duration - stage_time  # Last stage duration

        # Ensure that duration is positive
        if duration <= 0:
            raise ValueError(f"Invalid stage duration at stage {i}: duration={duration}")

        # Create the 9-tuple (duration, 8 rps)
        stage = (
            duration,
            merged_stages[i][1]["west_singlecore"],
            merged_stages[i][1]["west_multicore"],
            merged_stages[i][1]["east_singlecore"],
            merged_stages[i][1]["east_multicore"],
            merged_stages[i][1]["central_singlecore"],
            merged_stages[i][1]["central_multicore"],
            merged_stages[i][1]["south_singlecore"],
            merged_stages[i][1]["south_multicore"],
        )
        final_stages.append(stage)

    # Validate total duration
    calculated_total = sum(stage[0] for stage in final_stages)
    if calculated_total > total_duration:
        raise ValueError(f"Calculated total duration {calculated_total} exceeds the specified total_duration {total_duration}.")
    elif calculated_total < total_duration:
        # Add a final stage to fill the remaining duration with the last configuration
        last_stage = final_stages[-1]
        remaining_duration = total_duration - calculated_total
        final_stages.append((
            remaining_duration,
            last_stage[1],
            last_stage[2],
            last_stage[3],
            last_stage[4],
            last_stage[5],
            last_stage[6],
            last_stage[7],
            last_stage[8],
        ))

    return final_stages


from kubernetes import client, config

# Load Kubernetes configuration
config.load_kube_config()

# Kubernetes API client for deployments
apps_v1 = client.AppsV1Api()

# Define the namespace
namespace = "default"

# Define the image replacements
image_map = {
    "frontend": "docker.io/adiprerepa/boutique-frontend:latest",
    "checkout": "docker.io/adiprerepa/boutique-checkout:latest",
    "recommend": "docker.io/adiprerepa/boutique-recommendation:latest",
}

def update_first_container_image(deployment_name, new_image):
    # Get the current deployment
    deployment = apps_v1.read_namespaced_deployment(deployment_name, namespace)

    # Update the image of the first container
    if deployment.spec.template.spec.containers:
        deployment.spec.template.spec.containers[0].image = new_image
        # Apply the update
        apps_v1.patch_namespaced_deployment(deployment_name, namespace, deployment)
        print(f"Updated deployment '{deployment_name}' to image '{new_image}' for the first container")
    else:
        print(f"No containers found in deployment '{deployment_name}'")

def main():
    # List all deployments in the namespace
    deployments = apps_v1.list_namespaced_deployment(namespace)

    for deployment in deployments.items:
        deployment_name = deployment.metadata.name
        for prefix, new_image in image_map.items():
            if deployment_name.startswith(prefix):
                update_first_container_image(deployment_name, new_image)
                break

if __name__ == "__main__":
    main()

