import re
import sys
from collections import defaultdict

def parse_log_file(file_path):
    # Dictionary to store the traceId counts
    traceid_counts = defaultdict(int)
    
    # Regex pattern to capture the traceId (fifth entry in each log line)
    traceid_pattern = re.compile(r'\b[a-f0-9]{32}\b')

    with open(file_path, 'r') as file:
        for line in file:
            # Split the log line by spaces to get the entries
            entries = line.split()
            if len(entries) >= 5:
                # The fifth entry in each line is the traceId
                traceid_match = traceid_pattern.match(entries[4])
                if traceid_match:
                    traceid = entries[4]
                    traceid_counts[traceid] += 1

    # Check for duplicates
    duplicates = {traceid: count for traceid, count in traceid_counts.items() if count > 1}

    if duplicates:
        print("Duplicate traceIds found:")
        for traceid, count in duplicates.items():
            print(f"TraceId: {traceid}, Count: {count}")
    else:
        print("No duplicate traceIds found.")

# Example usage
logfile_path = sys.argv[1]  # Replace with the actual path to your log file
parse_log_file(logfile_path)