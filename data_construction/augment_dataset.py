import pandas as pd
import os
from PIL import Image

df = pd.read_json("../json_data/k_0_data.json")

image_names = set(
    os.listdir("../train_data")
)
df = df[df["img_filename"].isin(image_names)]

df = df.explode("elements").reset_index(drop=True)

df["instruction"] = df["elements"].apply(lambda x: x["instruction"])
df["bbox"] = df["elements"].apply(lambda x: x["bbox"])

df.drop(columns=["url", "elements"], inplace=True)


def create_new_points(row, index, k=4):
    img = Image.open(
        f'../train_data/{row["img_filename"]}'
    )
    w, h = img.size
    bbox = row["bbox"]
    instruction = row["instruction"]
    original_file_name = row["img_filename"]
    x1, y1, x2, y2 = bbox
    x, y = (x1 + x2) / 2, (y1 + y2) / 2
    x, y = x * w, y * h
    return_val = []
    img_crop = [0, 0, w, h]
    for i in range(1, k):
        w, h = img.size
        mid_w, mid_h = w / 2, h / 2
        new_x, new_y = x, y
        new_crop = img_crop
        # Determine the quadrant and adjust the bounds
        if x < mid_w and y < mid_h:  # Quadrant 4
            quadrant = 4
            new_crop = (0, 0, mid_w, mid_h)
        elif x >= mid_w and y < mid_h:  # Quadrant 1
            quadrant = 1
            new_crop = (mid_w, 0, w, mid_h)
            new_x -= mid_w  # Adjust x relative to the new image bounds
        elif x < mid_w and y >= mid_h:  # Quadrant 3
            quadrant = 3
            new_crop = (0, mid_h, mid_w, h)
            new_y -= mid_h  # Adjust y relative to the new image bounds
        else:  # x >= mid_w and y >= mid_h, Quadrant 2
            quadrant = 2
            new_crop = (mid_w, mid_h, w, h)
            new_x -= mid_w  # Adjust x relative to the new image bounds
            new_y -= mid_h  # Adjust y relative to the new image bounds

        # Original file name, saved file name, instruction, quadrant, point on this image, k
        split_name = original_file_name.split(".")
        new_name = split_name[0] + "-crop" + str(i) + f"row{index}" "." + split_name[1]
        new_point = (
            original_file_name,
            new_name,
            instruction,
            quadrant,
            (x / w, y / h),
            i,
        )
        return_val.append(new_point)
        path = os.path.join("../recursive_augmented_images", new_name)
        img.save(path)
        img = img.crop(new_crop)
        x, y = new_x, new_y

    return return_val


new_points = []
for i, row in df.iterrows():
    new_points.extend(create_new_points(row, i))

df = pd.DataFrame(
    new_points,
    columns=["img_filename", "new_name", "instruction", "quadrant", "point", "k"],
)

df.to_csv("data-recursive-main.csv")
