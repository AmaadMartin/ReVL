import pandas as pd
import json

K = 2
AUGMENTED_PER_POINT = 3

if __name__ == '__main__':
    # Load the CSV file to examine its structure
    csv_file_path = 'data-recursive-main.csv'
    data = pd.read_csv(csv_file_path)
    
    json_data = []
    index = 0
    for i in range(0, len(data), AUGMENTED_PER_POINT):
        row = data.iloc[i]
        if row['instruction'] == ' ' or row['instruction'] == '' or row['instruction'] == '.':
            continue
        for k in range(K):
            row = data.iloc[i + k]
            json_object = {
                "id": f"identity_{index}_{k}",
                "conversations": [
                    {
                        "from": "user",
                        "value": f"Picture 1: <img>data/recursive_augmented_images/{row['new_name']}</img>\n In this UI screenshot, what is the partition of the element corresponding to the command \"{row['instruction']}\" (with quadrant number)?"
                    },
                    {
                        "from": "assistant",
                        "value": str(row['quadrant'])
                    }
                ]
            }
            json_data.append(json_object)

        row = data.iloc[i + K]
        
        x = float(row['point'].split(',')[0].split('(')[1])
        y = float(row['point'].split(',')[1].split(')')[0])
        rounded_x = round(x, 2)
        rounded_y = round(y, 2)

        json_object = {
                        "id": f"identity_{index}_{K}",
                        "conversations": [
                            {
                                "from": "user",
                                "value": f"Picture 1: <img>data/recursive_augmented_images/{row['new_name']}</img>\n In this UI screenshot, what is the position of the element corresponding to the command \"{row['instruction']}\" (with point)?"
                            },
                            {
                                "from": "assistant",
                                "value": f"({rounded_x}, {rounded_y})",
                            }
                        ]
                    }
        json_data.append(json_object)
        index += 1

    with open(f'../json_data/k_{K}_data.json', 'w') as file:
        json.dump(json_data, file, indent=4)
