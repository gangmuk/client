import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

# Helper function to group by counter values that are within a specific range
def group_by_counter(df, tolerance=3):
    df = df.sort_values(by='counter')
    groups = []
    current_group = []

    for _, row in df.iterrows():
        if len(current_group) == 0:
            current_group.append(row)
        else:
            # Check if the current row's counter is close to the last row in the current group
            if abs(current_group[-1]['counter'] - row['counter']) <= tolerance:
                current_group.append(row)
            else:
                # If not, store the current group and start a new one
                groups.append(pd.DataFrame(current_group))
                current_group = [row]

        # If the group has 4 pods, finalize it and start a new one
        if len(current_group) == 4:
            groups.append(pd.DataFrame(current_group))
            current_group = []

    # Add any remaining group
    if len(current_group) > 0:
        groups.append(pd.DataFrame(current_group))

    return groups

# Load the CSV file
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python plot_hillclimb.py <file_path> <out_pdf>')
        sys.exit(1)
    
    file_path = sys.argv[1]
    out_pdf = sys.argv[2]
    df = pd.read_csv(file_path)

    # Strip any leading/trailing spaces from the column names
    df.columns = df.columns.str.strip()

    # Convert 'time' column to datetime for better plotting
    df['time'] = pd.to_datetime(df['time_millis']).astype('int64') // 10**3
    df['time'] -= df['time'].min()

    # Clip
    start_time = df['time'].min() + 0
    end_time = df['time'].max() - 0
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    
    # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Group by 'counter' values that are close to each other (within 2 or 3 of each other)
    grouped_data = group_by_counter(df, tolerance=3)

    # Calculate the average 'new-us-west-1' for each group of four pods
    avg_new_us_west_1 = []
    avg_times = []

    for group in grouped_data:
        avg_new_us_west_1.append(group['new-us-west-1'].mean())
        avg_times.append(group['time'].mean())  # Get the average time for the group

    # Plot the average new-us-west-1 for the groups
    ax1.plot(avg_times, avg_new_us_west_1, label='Grouped Average new-us-west-1', linestyle='-', marker='o', color='green')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Average West Local Serving Ratio', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.set_ylim(0, 1)
    
    ax1.legend(loc='lower right', fontsize='small')

    plt.title('Grouped Average New US West Serving Ratio Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure as a PDF
    plt.savefig(out_pdf)
