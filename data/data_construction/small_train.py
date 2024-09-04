import json
import random

# Set the number of random entries to select
K = 1
WITH_CONTEXT = True
JSON_PATH = '../json_data/{file_name}.json'
FILE_NAME = f'k_{K}_{"context_" if WITH_CONTEXT else ""}data'

if __name__ == '__main__':
    # Read the original dataset
    with open(JSON_PATH.format(file_name="seeclick_train_data_unaugmented"), 'r') as file:
        data = json.load(file)

    # get entries from random_indices.json
    with open('../json_data/test_indices.json', 'r') as file:
        random_indices = json.load(file)

    json_elements = []

    for i, index in enumerate(random_indices):
        # for k in range(K):
        #     quadrant_item = data[index*(K + 1) + k]
        #     quadrant_item['id'] = f"identity_{i}_{k}"
        #     json_elements.append(quadrant_item)

        # coordinate_item = data[index*(K + 1) + K]
        # coordinate_item['id'] = f"identity_{i}_{K}"
        # json_elements.append(coordinate_item)
        new_item = data[index]
        new_item['id'] = i
        json_elements.append(data[index])

    # Write the JSON string to a file
    with open(f'small_{FILE_NAME}.json', 'w', encoding='utf-8') as file:
        json.dump(json_elements, file, ensure_ascii=False, indent=4)