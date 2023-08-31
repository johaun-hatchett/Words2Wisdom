import json
import os
from argparse import ArgumentParser
from datetime import date

import gradio as gr
import tqdm
from nltk.tokenize import sent_tokenize

from pipeline import Text2KG


COOKBOOK = "./recipes.json"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--infile", type=str)
    parser.add_argument("--output", type=str, default="./output")
    # parser.add_argument("--cookbook", type=str, default=COOKBOOK,
    #                     help="path to prompt recipes")
    parser.add_argument("--recipe", type=str, choices=["Direct", "Traditional", "LogicBased"],
                        help="name of recipe to use"),
    # parser.add_argument("--thoughts", action="store_true",
    #                     help="whether to save GPT prompt/response chain")
    parser.add_argument("--demo", action="store_true",
                        help="execute Gradio app; overrides other arguments")
    
    return parser.parse_args()


def text2kg(recipe: str, text: str, progress=gr.Progress()):
    with open(COOKBOOK) as f:
        cookbook = json.load(f)
    
    for item in cookbook:
        if item["name"] == recipe:
            prompts = item
    
    pipe = Text2KG(prompts)
    sentences = sent_tokenize(text.replace("\n", " "))

    triplets = [pipe(s) for s in progress.tqdm(sentences, desc="Processing")]
    output = [{"sentence": s, "triplets": t} for s, t in zip(sentences, triplets)]

    return output


class App:
    def __init__(self):

        demo = gr.Interface(
            fn=text2kg,
            inputs=[
                gr.Radio(["Direct", "Traditional", "LogicBased"], label="Recipe"),
                gr.Textbox(lines=2, placeholder="Text Here...", label="Input Text")
            ],
            outputs=gr.JSON(label="KG Triplets"),
        )
        demo.queue(concurrency_count=10).launch()


def save(name, item, args):

    os.makedirs(args.output, exist_ok=True)

    today = date.today()
    filename = f"{today}_{name}_{args.recipe}.json"
    filepath = os.path.join(args.output, filename)

    with open(filepath, 'w') as f:
        json.dump(item, f)


def main(args):
    if args.demo:
        App()
    else:
        with open(args.infile) as f:
            text = f.read()

        output = text2kg(recipe=args.recipe, text=text, progress=tqdm)
        save("triplets", output, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)