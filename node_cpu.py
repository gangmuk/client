import paramiko
import matplotlib.pyplot as plt
import time
import threading
from datetime import datetime, timedelta
import subprocess


def start_node_cpu_monitoring(region_to_node, duration, filename, username="gangmuk", results_csv='cpu_utilization.csv'):
    # Run the collect_cpu_utilization function in a separate thread
    monitoring_thread = threading.Thread(
        target=collect_cpu_utilization, args=(region_to_node, username, duration, filename), kwargs={'results_csv': results_csv}
    )
    monitoring_thread.daemon = True
    monitoring_thread.start()
    return monitoring_thread


def collect_cpu_utilization(region_to_node, username, duration, filename, results_csv='cpu_utilization.csv'):
    # Set up SSH clients
    ssh_clients = {}
    for region, nodes in region_to_node.items():
        for node in nodes:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # Assuming SSH keys are set up
            ssh.connect(node, username=username)
            ssh_clients[node] = ssh

    # Initialize data storage
    cpu_data = {region: [] for region in region_to_node}
    cpu_data['loadgen-node'] = []
    timestamps = []

    # Function to get CPU usage from a node
    def get_cpu_usage(ssh_client):
        stdin, stdout, stderr = ssh_client.exec_command("top -bn1 | grep '%Cpu(s)'")
        output = stdout.read().decode('utf-8').strip()
        try:
            idle_percentage = float(output.split(",")[3].split()[0])
            utilization = 100 - idle_percentage
            return utilization
        except (IndexError, ValueError) as e:
            print(f"Error parsing CPU data: {e}")
            return None

    # Run for the specified duration
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=duration)
    
    while datetime.now() < end_time:
        elapsed_time = (datetime.now() - start_time).total_seconds()
        timestamps.append(elapsed_time)

        # Collect and aggregate data by region
        for region, nodes in region_to_node.items():
            region_utilization = []
            for node in nodes:
                ssh_client = ssh_clients.get(node)
                try:
                    cpu_usage = get_cpu_usage(ssh_client)
                    if cpu_usage is not None:
                        region_utilization.append(cpu_usage)
                except Exception as e:
                    print(f"Error collecting data from {node}: {e}")

            # Calculate average utilization for the region if data is available
            if region_utilization:
                avg_utilization = sum(region_utilization) / len(region_utilization)
                cpu_data[region].append(avg_utilization)
            else:
                cpu_data[region].append(None)

        # Collect CPU utilization for the local loadgen-node
        try:
            result = subprocess.run(["top", "-bn1"], stdout=subprocess.PIPE, text=True)
            output = result.stdout
            idle_percentage = float([line for line in output.splitlines() if '%Cpu(s)' in line][0].split(",")[3].split()[0])
            loadgen_cpu_usage = 100 - idle_percentage
            cpu_data['loadgen-node'].append(loadgen_cpu_usage)
        except (IndexError, ValueError, Exception) as e:
            print(f"Error collecting data from loadgen-node: {e}")
            cpu_data['loadgen-node'].append(None)

        time.sleep(4)  # Collect data every 5 seconds

    # Close SSH connections
    for ssh_client in ssh_clients.values():
        ssh_client.close()

    # Plot CPU utilization data by region
    for region, data in cpu_data.items():
        plt.plot(timestamps, data, label=region)
    
    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), loc="upper left")

    plt.xlabel("Seconds Elapsed")
    plt.ylabel("CPU Utilization (%)")
    plt.title("CPU Utilization by Region and Loadgen-Node Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot as a PDF
    plt.savefig(filename, format="pdf")
    print(f"CPU utilization graph saved as {filename}")
    # save to CSV File
    import csv
    with open(results_csv, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Seconds Elapsed'] + list(cpu_data.keys()))
        for i in range(len(timestamps)):
            writer.writerow([timestamps[i]] + [cpu_data[region][i] for region in cpu_data])

# Example usage:
# region_to_node = {
#     'us-west-1': ['node1', 'node2'],
#     'us-east-1': ['node3', 'node4'],
#     'us-central-1': ['node5', 'node6'],
#     'us-south-1': ['node7', 'node8']
# }
# collect_cpu_utilization(region_to_node, 'your_username', 60, 'cpu_utilization.pdf')
