import os
import re
import secrets
import string
import yaml
from datetime import datetime
from zipfile import ZipFile

import gradio as gr
import nltk
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from nltk.tokenize import sent_tokenize
from pandas import DataFrame

import utils
from chains import llm_chains


nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
stop_words = stopwords.words("english")


class Text2KG:
    """Text2KG class."""

    def __init__(self, api_key: str, **kwargs):

        self.model = ChatOpenAI(openai_api_key=api_key, **kwargs)
        self.embedding = OpenAIEmbeddings(openai_api_key=api_key)

    
    def init_pipeline(self, *steps: str):
        """Initialize Text2KG pipeline from passed steps.
        
        Args:
            *steps (str): Steps to include in pipeline. Must be a top-level name present in
                the schema.yml file
        """
        self.pipeline = SimpleSequentialChain(
            chains=[llm_chains[step](llm=self.model) for step in steps],
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


    def clean(self, kg: DataFrame):
        """Text2KG post-processing."""
        drop_list = []

        for i, row in kg.iterrows():
            # remove stopwords (pronouns)
            if (row.subject in stop_words) or (row.object in stop_words):
                drop_list.append(i)

            # remove broken triplets
            elif row.hasnans:
                drop_list.append(i)
            
            # lowercase nodes/edges, remove articles
            else:
                article_pattern = r'^(the|a|an) (.+)'
                be_pattern = r'^(are|is) (a )?(.+)'

                kg.at[i, "subject"] = re.sub(article_pattern, r'\2', row.subject.lower())
                kg.at[i, "relation"] = re.sub(be_pattern, r'\3', row.relation.lower())
                kg.at[i, "object"] = re.sub(article_pattern, r'\2', row.object.lower())

        return kg.drop(drop_list)


    def normalize(self, kg: DataFrame, threshold: float=0.3):
        """Reduce dimensionality of Text2KG output by merging cosine-similar nodes/edges."""

        ents = pd.concat([kg["subject"], kg["object"]]).unique()
        rels = kg["relation"].unique()

        ent_map = utils.condense_labels(ents, self.embedding.embed_documents, threshold=threshold)
        rel_map = utils.condense_labels(rels, self.embedding.embed_documents, threshold=threshold)

        kg_normal = pd.DataFrame()
        
        kg_normal["subject"] = kg["subject"].map(ent_map)
        kg_normal["relation"] = kg["relation"].map(rel_map)
        kg_normal["object"] = kg["object"].map(ent_map)

        return kg_normal


def extract_knowledge_graph(api_key: str, batch_size: int, modules: list[str], text: str, progress=gr.Progress()):
    """Extract knowledge graph from text.
    
    Args:
        api_key (str): OpenAI API key
        batch_size (int): Number of sentences per forward pass
        modules (list): Additional modules to add before main extraction step
        text (str): Text from which Text2KG will extract knowledge graph from
        progress: Progress bar. The default is gradio's progress bar; for a 
            command line progress bar, set `progress = tqdm`

    Returns:
        zip_path (str): Path to ZIP archive containing outputs
        knowledge_graph (DataFrame): The extracted knowledge graph
    """
    # init
    if api_key == "":
        raise ValueError("API key is required")
    
    model = Text2KG(api_key=api_key, temperature=0.3) # low temp. -> low randomness

    steps = []

    for module in modules:
        m = module.lower().replace(' ', '_')
        steps.append(m)

    if (len(steps) == 0) or (steps[-1] != "triplet_extraction"):
        steps.append("triplet_extraction")

    model.init_pipeline(*steps)

    # split text into batches
    sentences = sent_tokenize(text)
    batches = [" ".join(sentences[i:i+batch_size])
               for i in range(0, len(sentences), batch_size)]
    
    # create KG
    knowledge_graph = []
    
    for i, batch in progress.tqdm(list(enumerate(batches)), 
                                  desc="Processing...", unit="batches"):
        output = model.run(batch)
        [triplet.update({"sentence_id": i}) for triplet in output]

        knowledge_graph.extend(output)


    # convert to df, post-process data
    knowledge_graph = pd.DataFrame(knowledge_graph)
    knowledge_graph = model.clean(knowledge_graph)
    
    # rearrange columns
    knowledge_graph = knowledge_graph[["sentence_id", "subject", "relation", "object"]]

    # metadata
    now = datetime.now()
    date = str(now.date())

    metadata = {
        "_timestamp": now,
        "batch_size": batch_size,
        "modules": steps
    }

    # unique identifier for saving
    uid = ''.join(secrets.choice(string.ascii_letters)
                  for _ in range(6))
    
    save_dir = os.path.join(".", "output", date, uid)
    os.makedirs(save_dir, exist_ok=True)


    # save metadata & data
    with open(os.path.join(save_dir, "metadata.yml"), 'w') as f:
        yaml.dump(metadata, f)
    
    batches_df = pd.DataFrame(enumerate(batches), columns=["sentence_id", "text"])
    batches_df.to_csv(os.path.join(save_dir, "sentences.txt"), 
                     index=False)

    knowledge_graph.to_csv(os.path.join(save_dir, "kg.txt"), 
                           index=False)    
    

    # create ZIP file
    zip_path = os.path.join(save_dir, "output.zip")

    with ZipFile(zip_path, 'w') as zipObj:

        zipObj.write(os.path.join(save_dir, "metadata.yml"))
        zipObj.write(os.path.join(save_dir, "sentences.txt"))
        zipObj.write(os.path.join(save_dir, "kg.txt"))

    return zip_path, knowledge_graph


class App:
    def __init__(self):
        demo = gr.Interface(
            fn=extract_knowledge_graph,
            title="Text2KG",
            inputs=[
                gr.Textbox(placeholder="API key...", label="OpenAI API Key", type="password"),
                gr.Slider(minimum=1, maximum=10, step=1, label="Sentence Batch Size"),
                gr.CheckboxGroup(choices=["Clause Deconstruction"], label="Optional Modules"),
                gr.Textbox(lines=2, placeholder="Text Here...", label="Input Text"),
            ],
            outputs=[
                gr.File(label="Knowledge Graph"),
                gr.DataFrame(label="Preview", 
                             headers=["sentence_id", "subject", "relation", "object"], 
                             max_rows=10, 
                             overflow_row_behaviour="paginate")
            ],
            examples=[[None, 1, [], ("All cells share four common components: "
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
        demo.queue().launch(share=False)


if __name__ == "__main__":
    App()