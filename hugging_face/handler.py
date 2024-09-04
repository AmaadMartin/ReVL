from typing import Dict, List, Any
from peft import AutoPeftModelForCausalLM
import transformers
import os
import tempfile
from PIL import Image, ImageDraw
from io import BytesIO
import base64, json

COORDINATE_PROMPT = 'In this UI screenshot, what is the position of the element corresponding to the command \"{command}\" (with point)?'

PARTITION_PROMPT = 'In this UI screenshot, what is the partition of the element corresponding to the command \"{command}\" (with quadrant number)?'

class EndpointHandler():
    def __init__(self, path=""):
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    path,
                    device_map="cuda",
                    trust_remote_code=True,
                    fp16=True).eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    path,
                    cache_dir=None,
                    model_max_length=2048,
                    padding_side="right",
                    use_fast=False,
                    trust_remote_code=True,
                )
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        return
 
    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
       data args:
            image (:obj: `str`)
            task (:obj: `str`)
            k (:obj: `str`)
            context (:obj: 'str')
            kwargs
      Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """
        # open temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            image = os.path.join(temp_dir, "image.png")
            img = Image.open(BytesIO(base64.b64decode(data["inputs"]["image"])))
            img.save(image)
            command = data["inputs"]["task"]
            K = int(data["inputs"]["k"])
            keep_context = bool(data["inputs"]["context"])

            print(image)
            print(command)
            print(K)
            print(keep_context)

            images = [image]
            partitions = []

            for k in range(K):
                query = self.tokenizer.from_list_format(([{ 'image': context_image } for context_image in images] if keep_context else [{'image': image}]) + 
                [{'text': PARTITION_PROMPT.format(command=command)}])
                response, _ = self.model.chat(self.tokenizer, query=query, history=None)

                partition = int(response.split(" ")[-1])
                partitions.append(partition)

                # get cropped image of the partition
                with Image.open(image) as img:
                    width, height = img.size
                    if partition == 1:
                        img = img.crop((width // 2, 0, width, height // 2))
                    elif partition == 2:
                        img = img.crop((0, 0, width // 2, height // 2))
                    elif partition == 3:
                        img = img.crop((0, height // 2, width // 2, height))
                    elif partition == 4:
                        img = img.crop((width // 2, height // 2, width, height))
                    
                    new_path = os.path.join(temp_dir, f"partition{k}.png")
                    img.save(new_path)
                    image = new_path
                    images.append(image)
        
            query = self.tokenizer.from_list_format(([{ 'image': context_image } for context_image in images] if keep_context else [{'image': image}]) + 
            [{'text': COORDINATE_PROMPT.format(command=command)}])
            response, _ = self.model.chat(self.tokenizer, query=query, history=None)
            print("Coordinate Response:", response)

            x = float(response.split(",")[0].split("(")[1])
            y = float(response.split(",")[1].split(")")[0])

            for partition in partitions[::-1]:
                if partition == 1:
                    x = x/2 + 0.5
                    y = y/2
                elif partition == 2:
                    x = x/2
                    y = y/2
                elif partition == 3:
                    x = x/2
                    y = y/2 + 0.5
                elif partition == 4:
                    x = x/2 + 0.5
                    y = y/2 + 0.5
            print("rescaled point:", x, y)

            response = {}
            response['x'] = x
            response['y'] = y
            return response
            