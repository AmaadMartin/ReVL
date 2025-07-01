# %%
import json
import random
import os
from PIL import Image, ImageDraw

# Set the number of random entries to select
num_entries = 10000
# num_entries = 2

# Read the original dataset
with open('data/json_data/full_coordinate_train.json', 'r') as file:
    data = json.load(file)

len(data)

# %%
random_entries = random.sample(data, num_entries)
len(random_entries), random_entries[0]

# %%
K = 3

# %%
def ensure_folder_exists(folder_path):
   if not os.path.exists(folder_path):
       os.makedirs(folder_path)

def pick_quadrant(x_abs, y_asb, width, height):
    mid_w = width / 2
    mid_h = height / 2

    if x_abs < mid_w and y_asb < mid_h:
        return 2
    elif x_abs >= mid_w and y_asb < mid_h:
        return 1
    elif x_abs < mid_w and y_asb >= mid_h:
        return 3
    else:
        return 4
    

def crop_to_quadrant(img, quadrant, x_abs, y_abs):
    """
    Crops the PIL image 'img' to the selected quadrant (1..4).
    Also updates (x_abs, y_abs) to the new coordinate system.
    Returns (cropped_img, new_x_abs, new_y_abs).
    """
    w, h = img.size
    mid_w = w // 2
    mid_h = h // 2

    if quadrant == 1:
        # top-right quadrant
        crop_box = (mid_w, 0, w, mid_h)
        # offset: we subtract mid_w from x, and 0 from y
        x_offset = mid_w
        y_offset = 0
    elif quadrant == 2:
        # top-left quadrant
        crop_box = (0, 0, mid_w, mid_h)
        x_offset = 0
        y_offset = 0
    elif quadrant == 3:
        # bottom-left quadrant
        crop_box = (0, mid_h, mid_w, h)
        x_offset = 0
        y_offset = mid_h
    else:
        # quadrant 4: bottom-right
        crop_box = (mid_w, mid_h, w, h)
        x_offset = mid_w
        y_offset = mid_h

    cropped = img.crop(crop_box)

    # Update absolute coords
    new_x_abs = x_abs - x_offset
    new_y_abs = y_abs - y_offset

    return cropped, new_x_abs, new_y_abs
        

def draw_quadrant_lines(img):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # Red lines
    line_color = (0, 255, 0)
    line_width = 3

    # vertical line
    draw.line((w/2, 0, w/2, h), fill=line_color, width=line_width)
    # horizontal line
    draw.line((0, h/2, w, h/2), fill=line_color, width=line_width)
    return img  

# %%
def data_to_conversation(item, k, img_folder_path, lines=False):
    base_path = os.path.join('data/images', item["image"])
    base_img = Image.open(base_path)

    current_img = base_img
    w, h = current_img.size

    x_abs = item["coordinate"][0] * w
    y_abs = item["coordinate"][1] * h

    conversation = []

    original_image_name = item['image'].split('.')[0]
    for step_idx in range(k):
        if lines:
            step_filename = f"{original_image_name}_step{step_idx}_lines.png"
        else:
            step_filename = f"{original_image_name}_step{step_idx}.png"
        
        step_fullpath = os.path.join(img_folder_path, step_filename)

        if lines:
            draw_quadrant_lines(current_img)
        
        current_img.save(step_fullpath)

        user_prompt = (
            f"<img>{step_fullpath}</img>\n"
            f"In this UI screenshot, what is the partition of the element "
            f"corresponding to the command '{item['task']}' (with quadrant number)?"
        )

        quadrant = pick_quadrant(x_abs, y_abs, current_img.width, current_img.height)
        conversation.append({"from": "user", "value": user_prompt})
        conversation.append({"from": "assistant", "value": str(quadrant)})

        cropped, x_abs, y_abs = crop_to_quadrant(current_img, quadrant, x_abs, y_abs)
        current_img = cropped
    if lines:
        final_filename = f"{original_image_name}_step{k}_lines.png"
    else:
        final_filename = f"{original_image_name}_step{k}.png"
    
    final_fullpath = os.path.join(img_folder_path, final_filename)

    current_img.save(final_fullpath)

    user_prompt = (
        f"<img>{final_fullpath}</img>\n"
        f"In this UI screenshot, what is the position of the element corresponding "
        f"to the command '{item['task']}' (with point)?"
    )

    final_w, final_h = current_img.size
    x_norm =  round(x_abs / final_w, 2)
    y_norm = round(y_abs / final_h, 2)

    conversation.append({"from": "user", "value": user_prompt})
    conversation.append({"from": "assistant", "value": f"({x_norm}, {y_norm})"})

    return conversation
        
def data_to_conversation_quadrants(item, k, img_folder_path):
    base_img_path = os.path.join('data/images', item["image"])
    img = Image.open(base_img_path)
    w, h = img.size

    x_abs = item["coordinate"][0] * w
    y_abs = item["coordinate"][1] * h

    conversation = []

    for step_idx in range(k):
        quadrant_lines = []

        for q in [1, 2, 3, 4]:
            sub = crop_to_quadrant(img, q, x_abs, y_abs)[0]

            out_name = f"{item['image'].split('.')[0]}_q{q}_step{step_idx}.png"
            out_path = os.path.join(img_folder_path, out_name)
            sub.save(out_path)

            quadrant_lines.append(f"Quadrant {q}: <img>{out_path}</img>")
        
        user_prompt = (
            "\n".join(quadrant_lines) + "\n"
            + f"In this UI screenshot, what is the partition of the element "
            + f"corresponding to the command '{item['task']}' (with quadrant number)?"
        )

        conversation.append({"from": "user", "value": user_prompt})

        correct_quadrant = pick_quadrant(x_abs, y_abs, img.width, img.height)

        conversation.append({"from": "assistant", "value": str(correct_quadrant)})

        img, x_abs, y_abs = crop_to_quadrant(img, correct_quadrant, x_abs, y_abs)

    final_name = f"{item['image'].split('.')[0]}_step{k}_final.png"
    final_path = os.path.join(img_folder_path, final_name)
    img.save(final_path)

    user_prompt = (
        f"<img>{final_path}</img>\n"
        f"In this UI screenshot, what is the position of the element corresponding "
        f"to the command '{item['task']}' (with point)?"
    )

    conversation.append({"from": "user", "value": user_prompt})

    final_w, final_h = img.size
    x_norm =  round(x_abs / final_w, 2)
    y_norm = round(y_abs / final_h, 2)
    conversation.append({"from": "assistant", "value": f"({x_norm}, {y_norm})"})

    return conversation
    


# %%
def convert_all_items(data, k, img_folder_path, lines=False):
    conversations = []
    for i, item in enumerate(data):
        if i % 25 == 0:
            print(f"Processing item {i+1}/{len(data)}")
        conversation = data_to_conversation(item, k, img_folder_path, lines)
        conversations.append(conversation)
    
    return conversations

def convert_all_items_quadrant(data, k, img_folder_path):
    conversations = []
    for i, item in enumerate(data):
        if i % 25 == 0:
            print(f"Processing item {i+1}/{len(data)}")
        conversation = data_to_conversation_quadrants(item, k, img_folder_path)
        conversations.append(conversation)
    
    return conversations
print('======')
print("doing lines")
print('======')
line_conversations = convert_all_items(random_entries, K, 'data/experiment_images', lines=True)
print('======')
print("doing no lines")
print('======')
no_line_conversations = convert_all_items(random_entries, K, 'data/experiment_images', lines=False)
print('======')
print("doing quadrants")
print('======')
quadrant_conversations = convert_all_items_quadrant(random_entries, K, 'data/experiment_images')

with open('line_experiment_conversations.json', 'w') as file:
    json.dump(line_conversations, file)

with open('base_experiment_conversations.json', 'w') as file:
    json.dump(no_line_conversations, file)

with open('quadrant_experiment_conversations.json', 'w') as file:
    json.dump(quadrant_conversations, file)

# %%



