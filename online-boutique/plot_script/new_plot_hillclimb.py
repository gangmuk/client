import pandas as pd
import matplotlib.pyplot as plt
import sys

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
    
    temp = list()
    print(len(df))
    for i in range(len(df)):
        temp.append((i//4)*4*8)
    df["new_counter"] = temp

    print(f"len(df) = {len(df)}")
    print(f"len(temp) = {len(temp)}")
    print(f"len(temp) = {len(temp)}")
    print(f"len(df['time_millis']) = {len(df['time_millis'])}")
    
    # Convert 'time' column to datetime for better plotting
    # df['time'] = pd.to_datetime(df['time'])
    
    # df['time'] = pd.to_datetime(df['time']).astype('int64') // 10**9
    df['time'] = pd.to_datetime(df['time_millis']).astype('int64') // 10**3
    df['time'] -= df['time'].min()
    
    print(f"df['time']: {df['time']}")
    print(f"df['new_counter']: {df['new_counter']}")
    
    # Clip
    clip_front = 100
    clip_end = 30
    start_time = df['time'].min() + clip_front
    end_time = df['time'].max() - clip_end
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]

    grouped = df.groupby('podname')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ###################
    per_pod = True
    ###################
    
    ## Latency
    if per_pod == True:
        for podname, group in grouped:
            # ax1.plot(group['time'], group['avg_latency'], label=f'{podname} avg_latency')
            ax1.plot(group['time'], group['west_latency'], label=f'{podname} west_latency')
            ax1.plot(group['time'], group['east_latency'], label=f'{podname} east_latency')
            
    ## Inbound RPS        
    # if per_pod == True:
    #     for podname, group in grouped:
    #         ax1.plot(group['time'], group['inbound_rps'], label=f'{podname} west-inbound_rps', linestyle='-')
    # total_inbound_rps = df.groupby('new_counter')['inbound_rps'].sum()
    # ax1.plot(group['new_counter'], total_inbound_rps, label='total west-inbound_rps', linestyle='-', color='black')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Average Latency', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ## West local ratio
    if per_pod == True:
        for podname, group in grouped:
            # ax2.plot(group['time'], group['new-us-west-1'], label=f'{podname}-local routing', linestyle='--')
            ax2.plot(group['time'], group['new-us-east-1'], label=f'{podname}-offloading', linestyle='--')

    ax2.set_ylabel('New US West-1', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 1)
    ## East local ratio
    if per_pod == True:
        for podname, group in grouped:
            ax1.plot(group['time'], group['new-us-east-1'], label=f'{podname} new-us-east-1', linestyle=':')
    ## Average latency and average local ratio
    else:
        avg_latency_west = df.groupby('new_counter')['west_latency'].mean()
        avg_latency_east = df.groupby('new_counter')['east_latency'].mean()
        avg_new_us_west = df.groupby('new_counter')['new-us-west-1'].mean()
        avg_new_us_east = df.groupby('new_counter')['new-us-east-1'].mean()
        ax1.plot(avg_latency_west.index, avg_latency_west.values, label='Average West Latency', color='blue')
        ax1.plot(avg_latency_east.index, avg_latency_east.values, label='Average East Latency', color='red')
        ax2.plot(avg_new_us_west.index, avg_new_us_west.values, label='Average West US-West-1', color='green')
        ax2.plot(avg_new_us_east.index, avg_new_us_east.values, label='Average East US-East-1', color='orange')
    
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
    ax1.set_ylim(bottom=0)
    
    ########################
    # ax1.set_ylim(top=1000)
    ########################

    ax1.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=0)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    ax1.grid()
    ax2.grid()
    plt.grid()
    plt.title('New US East & West Latency and Pod-wise Average Latency Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_pdf)
