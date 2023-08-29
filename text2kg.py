from argparse import ArgumentParser
from datetime import date
import json
import os

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from pipeline import Text2KG


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("infile", type=str)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--cookbook", type=str, default="./recipes.json",
                        help="path to cookbook")
    parser.add_argument("--recipe", type=str, default=None,
                        help="name of recipe to use"),
    parser.add_argument("--thoughts", action="store_true")
    
    return parser.parse_args()


def save(name, item, args):

    os.makedirs(args.output, exist_ok=True)

    today = date.today()
    filename = f"{today}_{name}_{args.recipe}.json"
    filepath = os.path.join(args.output, filename)

    with open(filepath, 'w') as f:
        json.dump(item, f)


def main(args):
    with open(args.cookbook) as f:
        cookbook = json.load(f)
    
    recipe = None
    for item in cookbook:
        if item["name"] == args.recipe:
            recipe = item
    if recipe is None:
        raise ValueError(f"Recipe '{args.recipe}' does not exist in cookbook.")
        
    pipe = Text2KG(recipe)

    with open(args.infile) as f:
        text = f.read()

    sentences = sent_tokenize(text.replace('\n', ' '))
    
    triplets = [pipe(s) for s in tqdm(sentences)]

    output = [{"sentence": s, "triplets": t} for s, t in zip(sentences, triplets)]

    save("triplets", output, args)

    if args.thoughts:
        save("thoughts", pipe.history, args)
    
    return output


if __name__ == "__main__":
    args = parse_args()
    main(args)