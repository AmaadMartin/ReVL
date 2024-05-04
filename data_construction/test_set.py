import json
import random

JSON_PATH = '../json_data/seeclick_web_bbox.json'

if __name__ == '__main__':
    # Read the original dataset
    with open(JSON_PATH, 'r') as file:
        data = json.load(file)

    # get entries from random_indices.json
    with open('../json_data/test_indices.json', 'r') as file:
        test_indices = json.load(file)

    json_elements = []

    for i, index in enumerate(test_indices):
        test_point = data[index]
        json_elements.append(test_point)

    # Write the JSON string to a file
    with open(f'test_split.json', 'w', encoding='utf-8') as file:
        json.dump(json_elements, file, ensure_ascii=False, indent=4)