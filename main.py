import json
import os
import secrets
import string
from datetime import datetime
from zipfile import ZipFile

import gradio as gr
import pandas as pd
from langchain.chains import SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from nltk.tokenize import sent_tokenize

import utils
from chains import chains


class Text2KG:
    """Text2KG class."""

    def __init__(self, api_key: str, **kwargs):

        self.model = ChatOpenAI(openai_api_key=api_key, **kwargs)

    
    def init_pipeline(self, *steps: str):
        """Initialize Text2KG pipeline from passed steps.
        
        Args:
            *steps (str): Steps to include in pipeline. Must be a top-level name present in
                the schema.yml file
        """
        self.pipeline = SimpleSequentialChain(
            chains=[chains[step](llm=self.model) for step in steps],
            verbose=False
        )

    
    def run(self, text: str) -> list[dict]:
        """Run Text2KG pipeline on passed text.
        
        Arg:
            text (str): The text input
        
        Returns:
            triplets (list): The list of extracted KG triplets
        """
        triplets = self.pipeline.run(text)

        return triplets


def create_knowledge_graph(api_key: str, ngram_size: int, axiomatize: bool, text: str, progress=gr.Progress()):
    """Create knowledge graph from text.
    
    Args:
        api_key (str): OpenAI API key
        ngram_size (int): Number of sentences per forward pass
        axiomatize (bool): Whether to decompose sentences into simpler axioms as 
            a pre-processing step. Doubles the amount of calls to ChatGPT
        text (str): Text from which Text2KG will extract knowledge graph from
        progress: Progress bar. The default is gradio's progress bar; for a 
            command line progress bar, set `progress = tqdm`

    Returns:
        knowledge_graph (DataFrame): The extracted knowledge graph
        zip_path (str): Path to ZIP archive containing outputs
    """
    # init
    if api_key == "":
        raise ValueError("API key is required")
    
    model = Text2KG(api_key=api_key, temperature=0.3) # low temp. -> low randomness

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
    
    for i, ngram in progress.tqdm(enumerate(ngrams), desc="Processing...", total=len(ngrams)):
        output = model.run(ngram)
        [triplet.update({"sentence_id": i}) for triplet in output]

        knowledge_graph.extend(output)


    # convert to df, post-process data
    knowledge_graph = pd.DataFrame(knowledge_graph)
    knowledge_graph = utils.process(knowledge_graph)
    
    # rearrange columns
    knowledge_graph = knowledge_graph[["sentence_id", "subject", "relation", "object"]]

    # metadata
    now = datetime.now()
    date = str(now.date())
    timestamp = now.strftime("%Y%m%d%H%M%S")

    metadata = {
        "timestamp": timestamp,
        "batch_size": ngram_size,
        "axiom_decomposition": axiomatize
    }

    # unique identifier for saving
    uid = ''.join(secrets.choice(string.ascii_letters)
                  for _ in range(6))
    
    save_dir = os.path.join(".", "output", date, uid)
    os.makedirs(save_dir, exist_ok=True)


    # save metadata & data
    with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f)
    
    ngrams_df = pd.DataFrame(enumerate(ngrams), columns=["sentence_id", "text"])
    ngrams_df.to_csv(os.path.join(save_dir, "sentences.txt"), 
                     index=False)

    knowledge_graph.to_csv(os.path.join(save_dir, "kg.txt"), 
                           index=False)    
    

    # create ZIP file
    zip_path = os.path.join(save_dir, "output.zip")

    with ZipFile(zip_path, 'w') as zipObj:

        zipObj.write(os.path.join(save_dir, "metadata.json"))
        zipObj.write(os.path.join(save_dir, "sentences.txt"))
        zipObj.write(os.path.join(save_dir, "kg.txt"))

    return knowledge_graph, zip_path


class App:
    def __init__(self):
        description = (
            "# Text2KG\n\n"
            "Text2KG is a framework that uses ChatGPT to automatically creates knowledge graphs from plain text.\n\n"
            "**Usage:** (1) configure the pipeline; (2) add the text that will be processed"
        )
        demo = gr.Interface(
            fn=create_knowledge_graph,
            description=description,
            inputs=[
                gr.Textbox(placeholder="API key...", label="OpenAI API Key", type="password"),
                gr.Slider(minimum=1, maximum=10, step=1, label="Sentence Batch Size", info="Number of sentences per forward pass? Affects the number of calls made to ChatGPT.", ),
                gr.Checkbox(label="Axiom Decomposition", info="Decompose sentences into simpler axioms? (ex: \"I like cats and dogs.\" = \"I like cats. I like dogs.\")\n\nDoubles the number of calls to ChatGPT."),
                gr.Textbox(lines=2, placeholder="Text Here...", label="Input Text"),
            ],
            outputs=[
                gr.DataFrame(label="Knowledge Graph Triplets", 
                             headers=["sentence_id", "subject", "relation", "object"], 
                             max_rows=10, 
                             overflow_row_behaviour="show_ends"),
                gr.File(label="Knowledge Graph")
            ],
            examples=[["", 1, False, ("All cells share four common components: "
                                        "1) a plasma membrane, an outer covering that separates the "
                                        "cell's interior from its surrounding environment; 2) cytoplasm, "
                                        "consisting of a jelly-like cytosol within the cell in which "
                                        "there are other cellular components; 3) DNA, the cell's genetic "
                                        "material; and 4) ribosomes, which synthesize proteins. However, "
                                        "prokaryotes differ from eukaryotic cells in several ways. A "
                                        "prokaryote is a simple, mostly single-celled (unicellular) "
                                        "organism that lacks a nucleus, or any other membrane-bound "
                                        "organelle. We will shortly come to see that this is significantly "
                                        "different in eukaryotes. Prokaryotic DNA is in the cell's central "
                                        "part: the nucleoid.")]],
            allow_flagging="never",
            cache_examples=False
        )
        demo.launch(share=False)


if __name__ == "__main__":
    App()