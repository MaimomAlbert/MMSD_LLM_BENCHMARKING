import json
import csv

data_path = "/home/gpuuser3/sinngam_albert/datasets/osint/samples/vlm_selected.json"


with open(data_path, 'r') as json_file:
    data = json.load(json_file)

# Specify the keys you want to include in the CSV

selected_keys = data[0].keys()
print(selected_keys)

# print(selected_keys)

# Specify CSV file name
csv_file_name = '/home/gpuuser3/sinngam_albert/datasets/osint/samples/vlm_selected.csv'

# Write CSV file
with open(csv_file_name, 'w', newline='') as csv_file:
    # Create a CSV writer object
    csv_writer = csv.writer(csv_file)

    # Write header
    csv_writer.writerow(selected_keys)

    for row in data:
        selected_values = [row[key] for key in selected_keys]
        csv_writer.writerow(selected_values)

print(f'Conversion completed. CSV file saved as {csv_file_name}')
