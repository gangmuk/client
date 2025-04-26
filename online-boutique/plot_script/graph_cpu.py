import paramiko
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Nodes and login details
nodes = ["node1", "node2", "node3", "node4"]
username = "gangmuk"
# password = "your_password"  # Or use SSH keys for access.

# Interval between updates
interval = 5  # in seconds

# Initialize SSH clients for each node
ssh_clients = {}
for node in nodes:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(node, username=username)
    ssh_clients[node] = ssh

# Initialize data storage
cpu_data = {node: [] for node in nodes}
timestamps = []

def get_cpu_usage(ssh_client):
    stdin, stdout, stderr = ssh_client.exec_command("top -bn1 | grep '%Cpu(s)'")
    output = stdout.read().decode('utf-8').strip()
    # Extract the 'id' (idle) value
    try:
        idle_percentage = float(output.split(",")[3].split()[0])  # '99.2' in '99.2 id'
        utilization = 100 - idle_percentage  # Calculate utilization as 100 - idle
        print(utilization)
        return utilization
    except (IndexError, ValueError) as e:
        print(f"Error parsing CPU data: {e}")
        return None  # Return None if parsing fails

# Save frames to individual image files
frame_count = 0

while True:
    current_time = datetime.now().strftime("%H:%M:%S")
    timestamps.append(current_time)
    print(f"Collecting data at {current_time}")

    for node, ssh_client in ssh_clients.items():
        try:
            cpu_usage = get_cpu_usage(ssh_client)
            cpu_data[node].append(cpu_usage)
        except Exception as e:
            print(f"Error collecting data from {node}: {e}")
            cpu_data[node].append(None)

    if len(timestamps) > 20:  # Keep the last 20 timestamps to prevent overflow
        timestamps.pop(0)
        for data in cpu_data.values():
            data.pop(0)

    # Plotting setup
    plt.clf()
    for node, data in cpu_data.items():
        plt.plot(timestamps, data, label=node)
    
    plt.xlabel("Time")
    plt.ylabel("CPU Utilization (%)")
    plt.title("CPU Utilization Over Time")
    plt.legend(loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save each frame as a PNG image
    plt.savefig(f"cpu_utilization_frame_{frame_count}.png")
    print(f"Saved frame {frame_count} as cpu_utilization_frame_{frame_count}.png")
    frame_count += 1

    # Wait for the specified interval
    time.sleep(interval)
