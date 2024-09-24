import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_csv(csv_file, output_file):
    # Load the CSV data into a pandas DataFrame, handling quotes and commas properly
    df = pd.read_csv(csv_file, quotechar='"', skipinitialspace=True)

    # Filter the data for the primary y-axis (weight) based on the given conditions
    filtered_df_weight = df[(df['src_svc'] == 'sslateingress') & 
                            (df['dst_svc'] == 'frontend') & 
                            (df['src_cid'] == 'us-west-1') & 
                            (df['dst_cid'] == 'us-west-1')]

    # Filter the data for the secondary y-axis (flow) where dst_cid is 'us-west-1'
    filtered_df_flow_west = df[(df['src_svc'] == 'SOURCE') & 
                               (df['dst_svc'] == 'sslateingress') & 
                               (df['dst_cid'] == 'us-west-1')]

    # Filter the data for the secondary y-axis (flow) where dst_cid is 'us-east-1'
    filtered_df_flow_east = df[(df['src_svc'] == 'SOURCE') & 
                               (df['dst_svc'] == 'sslateingress') & 
                               (df['dst_cid'] == 'us-east-1')]

    # Merge the filtered DataFrames on the 'counter' column to align the data
    merged_df = pd.merge(filtered_df_weight[['counter', 'weight']],
                         filtered_df_flow_west[['counter', 'flow']],
                         on='counter', suffixes=('', '_west'))

    merged_df = pd.merge(merged_df,
                         filtered_df_flow_east[['counter', 'flow']],
                         on='counter', suffixes=('', '_east'))

    # Plotting
    fig, ax1 = plt.subplots()

    # Plot weight on the primary y-axis
    ax1.set_xlabel('Counter')
    ax1.set_ylabel('Weight', color='tab:blue')
    ax1.plot(merged_df['counter'], merged_df['weight'], color='tab:blue', label='Weight')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a secondary y-axis for the flow columns
    ax2 = ax1.twinx()
    ax2.set_ylabel('Flow', color='tab:green')

    # Plot flow for dst_cid = 'us-west-1'
    ax2.plot(merged_df['counter'], merged_df['flow'], color='tab:green', label='Flow (us-west-1)')
    
    # Plot flow for dst_cid = 'us-east-1'
    ax2.plot(merged_df['counter'], merged_df['flow_east'], color='tab:orange', label='Flow (us-east-1)')
    
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Set the title of the plot
    plt.title('Counter vs Weight and Flow (West and East)')

    # Add the legend to label the lines
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the plot as a PDF
    plt.savefig(output_file, format='pdf')

    # Show the plot (optional)
    # plt.show()

if __name__ == "__main__":
    # First argument is the CSV file, second argument is the output PDF file
    csv_file = sys.argv[1]
    output_file = sys.argv[2]
    
    plot_csv(csv_file, output_file)
