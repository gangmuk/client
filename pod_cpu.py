import time
import threading
import matplotlib.pyplot as plt
import csv
from kubernetes import client, config
from datetime import datetime, timedelta
from kubernetes.client import CustomObjectsApi

def graph_pod_cpu_utilization(deployments, regions, namespace, duration, pdf_file, interval=5):
    """
    Collects CPU utilization of all pods for the given deployments over the specified duration and graphs it.

    :param deployments: List of deployment names to monitor.
    :param regions: List of regions corresponding to each deployment.
    :param namespace: Namespace of the deployments.
    :param pdf_file: File path to save the graph as a PDF.
    :param duration: Total duration in seconds to collect data.
    :param interval: Interval in seconds between data collection points.
    """
    # Configure access to the Kubernetes cluster
    config.load_kube_config()
    v1_apps = client.AppsV1Api()
    metrics_api = CustomObjectsApi()

    start_time = datetime.now()
    print("duration and type of duration", duration, type(duration))
    end_time = start_time + timedelta(seconds=int(duration))
    cpu_data = {f"{deployment}-{region}": [] for deployment in deployments for region in regions}
    timestamps = []

    while datetime.now() < end_time:
        current_time = datetime.now().strftime('%H:%M:%S')
        timestamps.append(current_time)

        for deployment in deployments:
            for region in regions:
                deployment_key = f"{deployment}-{region}"
                try:
                    # Get the pods of the given deployment with matching region
                    label_selector = f"app={deployment},region={region}"
                    # print(f"Fetching pods for deployment '{deployment}' with label selector '{label_selector}'")
                    pods = client.CoreV1Api().list_namespaced_pod(namespace, label_selector=label_selector).items

                    # Aggregate CPU usage across all pods of the deployment (only the primary container)
                    total_cpu_usage = 0
                    for pod in pods:
                        pod_name = pod.metadata.name
                        try:
                            # print(f"Fetching metrics for pod '{pod_name}'")
                            # Fetching metrics from metrics.k8s.io API (make sure Metrics Server is running)
                            metrics = metrics_api.get_namespaced_custom_object(
                                group="metrics.k8s.io",
                                version="v1beta1",
                                namespace=namespace,
                                plural="pods",
                                name=pod_name
                            )
                            # Only consider the first container (primary container)
                            container = next((c for c in metrics['containers'] if c['name'] != 'istio-proxy'), None)
                            if container:
                                cpu_usage = container['usage']['cpu']
                                if cpu_usage.endswith('n'):  # Nanocores to cores conversion
                                    total_cpu_usage += int(cpu_usage.rstrip('n')) / 1e9
                                elif cpu_usage.endswith('m'):  # Millicores to cores conversion
                                    total_cpu_usage += int(cpu_usage.rstrip('m')) / 1000
                                # print(f"Pod '{pod_name}' CPU usage: {cpu_usage}")
                            else:
                                print(f"No valid container found for pod '{pod_name}'")
                        except Exception as e:
                            print(f"Error fetching metrics for pod {pod_name}: {e}")

                    cpu_data[deployment_key].append(total_cpu_usage)
                    # print(f"Total CPU usage for deployment '{deployment}' in region '{region}' at {current_time}: {total_cpu_usage}")
                except Exception as e:
                    print(f"Error fetching pods for deployment {deployment}: {e}")

        time.sleep(interval)

    # Save CPU metrics to CSV
    csv_file = pdf_file + ".csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Timestamp"] + list(cpu_data.keys()))
        # Write data rows
        for i, timestamp in enumerate(timestamps):
            row = [timestamp] + [cpu_data[deployment_key][i] if i < len(cpu_data[deployment_key]) else '' for deployment_key in cpu_data.keys()]
            writer.writerow(row)

    # Plotting the CPU utilization, with separate lines for each deployment-region combination
    plt.figure(figsize=(10, 6))
    plt.title('Aggregate CPU Utilization for Deployments')
    plt.xlabel('Time')
    plt.ylabel('CPU Usage (cores)')

    for deployment_key, cpu_usages in cpu_data.items():
        app, region = deployment_key.split("-", 1)
        plt.plot(timestamps[:len(cpu_usages)], cpu_usages, label=f'{app} ({region})')

    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(pdf_file)
    plt.close()
