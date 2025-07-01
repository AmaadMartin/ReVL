from PIL import Image
import json
# from IPython.display import display
# from tqdm import tqdm

if __name__ == "__main__":
    # preprocess images for ReVL
    data_path = "../json_data/ReVL_text_to_point_20000.json"
    with open(data_path, "r") as file:
        data = json.load(file)

    errored = []
    length = len(data)
    for i in range(length):
        print(str((i * 100)/length) + "%")
        entry = data[i]
        if entry["type"] == "revl":
            for step in entry["conversations"]:
                if step["from"] == "user":
                    try:
                        zoom = step["zoom"]
                        img_path = step["Image"]
                        # print(img_path)
                        # print(zoom) 
                        img = Image.open("../../" + img_path)
                        width, height = img.size
                        # zoom is a normalized bounding box in the format (left, upper, right, lower)
                        left = int(zoom[0] * width)
                        upper = int(zoom[1] * height)
                        right = int(zoom[2] * width)
                        lower = int(zoom[3] * height)
                        zoomed_img = img.crop((left, upper, right, lower))
                        # save the zoomed in image with zoom coordinates in the filename
                        # check if the image is a PNG or JPG
                        if ".jpg" in img_path:
                            new_img_path = img_path.replace(".jpg", f"_{left}_{upper}_{right}_{lower}.jpg")
                        else:
                            new_img_path = img_path.replace(".png", f"_{left}_{upper}_{right}_{lower}.png")
                        # show before and after images
                        # display(img)
                        # display(zoomed_img)
                        # print(new_img_path)
                        zoomed_img.save("../../" + new_img_path)
                        step["value"] = f"Picture 1: <img>{new_img_path}</img>\n" + step["value"]
                        # print(step["value"])
                    except Exception as e:
                        errored.append((i, img_path, e))
                        print(e)
                        print("Error in image: ", img_path)
                        print("width: ", width)
                        print("height: ", height)
                        print("zoom: ", zoom)
                        print("unnormalized zoom: ", (left, upper, right, lower))
                        continue


    print(errored)
    # save the updated data to a new JSON file
    updated_data_path = data_path.replace(".json", "_with_augmented_images.json")
    with open(updated_data_path, "w") as file:
        json.dump(data, file, indent=4)