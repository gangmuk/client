import subprocess
import json
from collections import defaultdict

# AWS inter-region data transfer costs (USD per GB)
AWS_EGRESS_COSTS = {
    "us-west-1": {  # Northern California
        "us-east-1": 0.02,     # N. Virginia
        "us-central-1": 0.02,  # US Central (Iowa)
        "us-south-1": 0.02     # US South (Texas)
    },
    "us-east-1": {   # N. Virginia
        "us-west-1": 0.02,     # N. California
        "us-central-1": 0.02,  # US Central (Iowa)
        "us-south-1": 0.02     # US South (Texas)
    },
    "us-central-1": { # US Central (Iowa)
        "us-west-1": 0.02,     # N. California
        "us-east-1": 0.02,     # N. Virginia
        "us-south-1": 0.02     # US South (Texas)
    },
    "us-south-1": {  # US South (Texas)
        "us-west-1": 0.02,     # N. California
        "us-east-1": 0.02,     # N. Virginia
        "us-central-1": 0.02   # US Central (Iowa)
    }
}

def setup_tc_byte_counters(network_interface, node_dict, inter_cluster_latency):
    """Sets up tc filters with byte counters for each source-destination node pair"""
    for src_node, dst_nodes in inter_cluster_latency.items():
        for dst_node in dst_nodes:
            dst_ip = node_dict[dst_node]['ipaddr']
            
            # Add filter with byte counter as an action, using high priority number 
            # to ensure it runs after existing filters
            cmd = f"sudo tc filter add dev {network_interface} parent 1: protocol ip prio 100 \
                   u32 match ip dst {dst_ip}/32 action gact"
            try:
                subprocess.run(cmd.split(), check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to add filter for {dst_ip}: {e.stderr.decode()}")

def get_byte_counts(network_interface, node_dict, inter_cluster_latency):
    """Retrieves byte counts from tc filters for all node pairs"""
    byte_counts = defaultdict(lambda: defaultdict(int))
    
    # Get all filter statistics at once
    cmd = f"sudo tc -s filter show dev {network_interface} parent 1:"
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
        current_filter = None
        current_bytes = None
        
        for line in result.stdout.split('\n'):
            # Look for lines containing IP addresses
            for src_node, dst_nodes in inter_cluster_latency.items():
                for dst_node in dst_nodes:
                    dst_ip = node_dict[dst_node]['ipaddr']
                    if f"match {dst_ip}/32" in line:
                        current_filter = (src_node, dst_node)
                    
            # Look for bytes count in the statistics
            if current_filter and 'Sent' in line:
                try:
                    bytes_sent = int(line.split()[1])
                    src_node, dst_node = current_filter
                    byte_counts[src_node][dst_node] = bytes_sent
                except (IndexError, ValueError):
                    pass
                current_filter = None
                
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to get filter statistics: {e.stderr.decode()}")
        
    return byte_counts

def calculate_egress_costs(byte_counts, node_to_region):
    """
    Calculates egress costs based on byte counts and AWS region-specific pricing
    Only counts traffic between different regions
    """
    costs = defaultdict(lambda: defaultdict(float))
    
    for src_node, dst_counts in byte_counts.items():
        src_region = node_to_region[src_node]
        for dst_node, bytes_sent in dst_counts.items():
            dst_region = node_to_region[dst_node]
            
            if src_region != dst_region:  # Only count inter-region traffic
                if src_region in AWS_EGRESS_COSTS and dst_region in AWS_EGRESS_COSTS[src_region]:
                    cost_per_gb = AWS_EGRESS_COSTS[src_region][dst_region]
                    gb_sent = bytes_sent / (1024 * 1024 * 1024)  # Convert to GB
                    costs[src_region][dst_region] = gb_sent * cost_per_gb
                
    return costs

def remove_byte_counters(network_interface):
    """Removes only our byte counter filters without affecting other rules"""
    cmd = f"sudo tc filter show dev {network_interface} parent 1:"
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
        
        # Find and delete only our filters (priority 100)
        for line in result.stdout.split('\n'):
            if "prio 100" in line:
                # Extract filter handle
                try:
                    handle = line.split()[3].rstrip(':')
                    delete_cmd = f"sudo tc filter del dev {network_interface} parent 1: handle {handle} prio 100 protocol ip"
                    subprocess.run(delete_cmd.split(), check=True)
                except (IndexError, subprocess.CalledProcessError):
                    continue
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to remove filters: {e.stderr.decode()}")

def print_traffic_summary(costs):
    """Prints a human-readable summary of traffic costs"""
    total_cost = 0
    print("\nTraffic Summary:")
    print("-" * 50)
    for src_region, dst_costs in costs.items():
        for dst_region, cost in dst_costs.items():
            print(f"{src_region} -> {dst_region}: ${cost:.2f}")
            total_cost += cost
    print("-" * 50)
    print(f"Total egress cost: ${total_cost:.2f}")