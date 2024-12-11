import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    # Check if both CSV file and output PDF paths are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv_path> <output_pdf_path>")
        sys.exit(1)

    # Get file paths from command line arguments
    csv_file = sys.argv[1]
    output_pdf = sys.argv[2]

    # Add .pdf extension if not present
    if not output_pdf.endswith('.pdf'):
        output_pdf += '.pdf'

    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S')
        
        # Convert time to seconds from start
        start_time = df['Timestamp'].min()
        df['Seconds'] = (df['Timestamp'] - start_time).dt.total_seconds()

        # Create the plot with adjusted figure size
        plt.figure(figsize=(12, 5))

        # Plot lines for each region with specific colors and larger linewidth
        plt.plot(df['Seconds'], df['frontend-us-west-1'], label='Oregon', color='#4285F4', linewidth=4)
        plt.plot(df['Seconds'], df['frontend-us-central-1'], label='Iowa', color='#DB4437', linewidth=4)
        plt.plot(df['Seconds'], df['frontend-us-south-1'], label='Utah', color='#F4B400', linewidth=4)
        plt.plot(df['Seconds'], df['frontend-us-east-1'], label='S. Carolina', color='#0F9D58', linewidth=4)

        # Customize the plot
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Time (s)', fontsize=24, fontweight='bold')
        plt.ylabel('CPU Util (Normalized)', fontsize=24, fontweight='bold')
        plt.title('Regional CPU Utilization Over Time for Frontend', fontsize=24, fontweight='bold', pad=20)

        # Set y-axis to start at 0
        plt.ylim(bottom=0)

        # Customize legend with larger font
        plt.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1, 1))

        # Customize tick labels with larger font
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save as PDF with high DPI for quality
        plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Graph saved as {output_pdf}")

    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()