import os
import re
from time import time
from argparse import ArgumentParser
from typing import List
from zipfile import ZipFile

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from . import VALIDATION_CONFIG, OUTPUT_DIR
from .output_parsers import QuestionOutputParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "run_ids", nargs="+", help="Run IDs to evaluate. Format: YYYY-MM-DD-XXX"
    )
    parser.add_argument(
        "--search_dir", help="Directory to search for output", default=OUTPUT_DIR
    )
    return parser.parse_args()


def format_system_prompt():
    """Format instructional prompt from instructions.yml"""

    def format_question(question: dict):
        # Question format: {text} {additional} {options}
        # Ex. Can pigs fly? Explain. (Yes/No)
        formatted = (
            question["title"]
            + " "
            + question["text"]
            + " "
            + (question["additional"] + " " if question["additional"] else "")
            + "("
            + ";".join(question["options"])
            + ")\n"
        )
        return formatted

    def format_example(example: dict):
        formatted = (
            f"PASSAGE: {example['passage']}\n"
            f"TRIPLET: {example['triplet']}\n\n"
            + "".join([
                f"{i}) {answer}\n"
                for i, answer in enumerate(example["answers"].values(), start=1)
            ])
        )
        return formatted

    instruction = VALIDATION_CONFIG["instruction"]
    questions = "".join([
        f"{i}) {format_question(q)}"
        for i, q in enumerate(VALIDATION_CONFIG["questions"].values(), start=1)
    ])
    example = format_example(VALIDATION_CONFIG["example"])

    system_prompt = (
        f"{instruction}\n" f"QUESTIONS:\n{questions}\n" f"[* EXAMPLE *]\n\n{example}"
    )

    return system_prompt


def validate_triplets(
    llm: ChatOpenAI, instruction: str, passage: str, triplets: List[List[str]]
) -> List[pd.DataFrame]:
    """Validate triplets with respect to passage."""

    print(
        f"Validating {len(triplets):>2} triplet{'s' if len(triplets) else ''}...",
        end=" ",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{instruction}"),
        ("user", "PASSAGE: {passage}\n\nTRIPLET: {triplet}"),
    ])

    chain = prompt | llm | QuestionOutputParser()

    output = chain.batch([
        {"instruction": instruction, "passage": passage, "triplet": triplet}
        for triplet in triplets
    ])

    print("Done!", end="\n")
    return output


def validate_knowledge_graph(llm: ChatOpenAI, output_zip: str):
    """Validate all triplets in a knowledge graph."""

    run_id = re.findall(r"output-(.*)\.zip", output_zip)[0]
    run_dir = os.path.dirname(output_zip)

    # read output zip
    with ZipFile(output_zip) as z:
        # load knowledge graph
        with z.open("kg.csv") as f:
            graph = pd.read_csv(f)

        # load text batches
        with z.open("text_batches.csv") as f:
            text_batches = pd.read_csv(f)

    print("Initializing knowledge graph validation. Run:", run_id)
    print()

    # start stopwatch
    start = time()

    # container for evaluation responses
    responses = []

    # instructions
    instruction = format_system_prompt()

    # triplets are batched by passage
    # so we iterate over passages
    for idx, passage in enumerate(text_batches.text):
        triplets = (
            graph[graph["batch_id"] == idx].drop(columns=["batch_id"]).values.tolist()
        )

        print(f"Starting excerpt {idx + 1:>2} of {len(text_batches)}.", end=" ")

        # if batch has no triplets to validate, skip batch
        if len(triplets) == 0:
            print("Excerpt has no triplets to validate.", end="\n")
            continue

        # validate triplets by batch
        response = validate_triplets(
            llm=llm, 
            instruction=instruction, 
            passage=passage, 
            triplets=triplets
        )
        responses.extend(response)

    validation = pd.concat(responses, ignore_index=True)

    # merge with knowlege graph data
    validation_merged = (
        text_batches.merge(graph)
        .drop(columns=["batch_id"])
        .merge(validation, left_index=True, right_index=True)
    )

    savepath = os.path.join(run_dir, f"validation-{run_id}.csv")
    validation_merged.to_csv(savepath, index=False)
    # stop stopwatch
    end = time()

    print("\nKnowledge graph validation complete!")
    print(f"It took {end - start:0.3f} seconds to validate {len(validation)} triplets.")
    print("Saved to:", savepath)

    return savepath


if __name__ == "__main__":
    args = parse_args()

    llm = ChatOpenAI(
        model="gpt-4-turbo-preview", 
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    for run_id in args.run_ids:
        zipfile = os.path.join(args.search_dir, f"output-{run_id}.zip")
        validate_knowledge_graph(llm, run_id)
        print("* " * 25)
