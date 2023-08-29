import re
import time

from chatbot import ChatGPT


class BasePipeline:
    
    chat = ChatGPT(model="gpt-3.5-turbo")

    def add_task(self, task, **inputs):
        args = task["args"]
        args_supplied = {arg: arg in inputs for arg in args}
        
        if all(args_supplied.values()):

            formatted_task = f"[TASK]\n{task['description']}"

            if len(args):
                formatted_task += f"\n\n[INPUT]\n" + '\n'.join([f'{arg}: {inputs[arg]}' for arg in args])

            self.pipe.append(formatted_task)
        else:
            raise ValueError(
                f"Missing required argument(s): {[arg for arg in args if not args_supplied[arg]]}"
            )
    

    def compile_tasks(self, inputs_by_idx):
        self.pipe = []

        for idx, task in enumerate(self.prompt):
            self.add_task(task, **inputs_by_idx.get(idx, {}))
    
    
    def forward(self):
        self.history = []
        
        for task in self.pipe:
            output = self.chat(task)
            time.sleep(0.2)

        self.history.append(self.chat.history)
        self.chat.clear_history()
        
        return self.postprocess(output)
    

    def postprocess(self):
        raise NotImplementedError


class Text2KG(BasePipeline):

    
    def __init__(self, recipe: dict):

        self.name = recipe["name"]
        self.prompt = recipe["prompt"]

        self.chat = ChatGPT(
            model="gpt-3.5-turbo",
            init=(
                "You are a sentence parsing agent helping to construct a knowledge graph."
            ),
            temperature=0.3
        )

    
    def __call__(self, text):
        self.compile_tasks(
            inputs_by_idx={0: {"text": text}}
        )

        return self.forward()
    

    def __repr__(self):
        return f"Text2KG(recipe={self.name})"

    
    def postprocess(self, output: str):
        
        word_pattern = r"'?\w+(?:[ |.'-]\w+)*'?"
        triplet_pattern = f'({word_pattern}::{word_pattern}::{word_pattern})'
        processed = re.findall(triplet_pattern, re.sub(r'[<>]', '', output))
        
        return processed
