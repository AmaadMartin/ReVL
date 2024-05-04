import json
import random

# Set the number of random entries to select
K = 2
JSON_PATH = '../json_data/{file_name}.json'
FILE_NAME = f'k_{K}_data'

if __name__ == '__main__':
    # Read the original dataset
    with open(JSON_PATH.format(file_name=FILE_NAME), 'r') as file:
        data = json.load(file)

    # get entries from random_indices.json
    with open('../json_data/random_indices.json', 'r') as file:
        random_indices = json.load(file)

    json_elements = []

    for i, index in enumerate(random_indices):
        for k in range(K):
            quadrant_item = data[index*(K + 1) + k]
            quadrant_item['id'] = f"identity_{i}_{k}"
            json_elements.append(quadrant_item)

        coordinate_item = data[index*(K + 1) + K]
        coordinate_item['id'] = f"identity_{i}_{K}"
        json_elements.append(coordinate_item)

    # Write the JSON string to a file
    with open(f'small_{FILE_NAME}.json', 'w', encoding='utf-8') as file:
        json.dump(json_elements, file, ensure_ascii=False, indent=4)