import os
from datetime import datetime

import gradio as gr
import pandas as pd
from langchain.chains import SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from nltk.tokenize import sent_tokenize

from chains import chains
from process import process


class Text2KG:

    def __init__(self, api_key: str, **kwargs):

        self.model = ChatOpenAI(openai_api_key=api_key, **kwargs)

    
    def init_pipeline(self, *steps: str):
        self.pipeline = SimpleSequentialChain(
            chains=[chains[step](llm=self.model) for step in steps],
            verbose=False
        )

    
    def run(self, text: str):
        triplets = self.pipeline.run(text)

        [triplet.update({"context": text}) for triplet in triplets]

        return  triplets


def create_knowledge_graph(api_key: str, ngram_size: int, axiomatize: bool, text: str, progress=gr.Progress()):

    # init
    if api_key == "":
        raise ValueError("API key is required")
    
    model = Text2KG(api_key=api_key, temperature=0.3)

    if axiomatize:
        steps  = ["text2axiom", "extract_triplets"]
    else:
        steps = ["extract_triplets"]

    model.init_pipeline(*steps)

    # split text into ngrams
    sentences = sent_tokenize(text)
    ngrams = [" ".join(sentences[i:i+ngram_size]) 
              for i in range(0, len(sentences), ngram_size)]
    
    # create KG
    knowledge_graph = []
    
    for ngram in progress.tqdm(ngrams, desc="Processing..."):
        output = model.run(ngram)
        knowledge_graph.extend(output)

    knowledge_graph = pd.DataFrame(knowledge_graph)
    knowledge_graph = process(knowledge_graph)


    now = datetime.now()
    date = str(now.date())
    timestamp = now.strftime("%Y%m%d%H%M%S")
    
    path = os.path.join(".", "output", date)
    os.makedirs(path, exist_ok=True)
    
    filename = f"kg-{timestamp}--batch-{ngram_size}--axiom-{axiomatize}.csv"
    filepath = os.path.join(path, filename)

    knowledge_graph.to_csv(filepath, index=False)

    return knowledge_graph, filepath


class App:
    def __init__(self):
        description = (
            "Text2KG is a framework that creates knowledge graphs from unstructured text.\n"
            "The framework uses ChatGPT to fulfill this task.\n"
            "First, configure the pipeline, then add the text that will be processed."
        )
        demo = gr.Interface(
            fn=create_knowledge_graph,
            description=description,
            inputs=[
                gr.Textbox(placeholder="API key...", label="OpenAI API Key"),
                gr.Slider(maximum=10, step=1, label="Batching", info="Number of sentences per batch? (0 = do not chunk text)", ),
                gr.Checkbox(label="Axiomatize", info="Decompose sentences into simpler axioms?\n(ex: \"I like cats and dogs.\" = \"I like cats. I like dogs.\")"),
                gr.Textbox(lines=2, placeholder="Text Here...", label="Input Text"),
            ],
            outputs=[
                gr.DataFrame(label="Knowledge Graph Triplets", 
                             headers=["subject", "relation", "object", "context"], 
                             max_rows=10, 
                             overflow_row_behaviour="show_ends"),
                gr.File(label="Knowledge Graph")
            ],
            examples=[["", 1, True, description]],
            allow_flagging="never"
        )
        demo.queue(concurrency_count=10).launch(share=False)


if __name__ == "__main__":
    App()