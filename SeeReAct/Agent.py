from openai import OpenAI
import os
from dotenv import load_dotenv
from Models import OutputFormat
from Prompt import agent_prompt, query_prompt
import base64
import tempfile

load_dotenv()

class Agent:
    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.actions = []

    def act(self, screen, task):
        # expects screen to be a PIL Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            screen.save(temp.name)
            temp_path = temp.name
            encoded_screen = self.encode_image(temp_path)
            messages = self.create_messages(encoded_screen, task)

            response = self.openai.beta.chat.completions.parse(
                model="gpt-4o", messages=messages, response_format=OutputFormat
            )

            action = response.choices[0].message.parsed
            self.actions.append(action.json())
            return action

    def reset(self):
        self.actions = []

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def create_messages(self, base64_image, task):
        messages = [
            {"role": "system", "content": agent_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query_prompt.format(task=task, actions=self.actions),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ]
        return messages
