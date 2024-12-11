import csv
import sys

def expand_csv(input_file, output_file):
    # Regions to add as a new column
    regions = ['us-west-1', 'us-east-1', 'us-central-1', 'us-south-1']
    
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            for row in reader:
                for region in regions:
                    # Prepend region to the row
                    new_row = [region] + row
                    writer.writerow(new_row)
                    
        print(f"CSV expanded successfully. Output saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    expand_csv(input_csv, output_csv)
