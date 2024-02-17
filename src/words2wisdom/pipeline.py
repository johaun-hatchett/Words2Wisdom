import re
from typing import List

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from nltk.tokenize import sent_tokenize

from . import MODULES_CONFIG, STOP_WORDS
from .config import Config
from .output_parsers import ClauseParser, TripletParser
from .utils import partition_sentences


# llm output parsers
PARSERS = {
    "StrOutputParser": StrOutputParser(),
    "ClauseParser": ClauseParser(),
    "TripletParser": TripletParser()
}


class Module:
    """Text2KG module class."""
    def __init__(self, name: str) -> None:
        self.name = name
        self.parser = self.get_parser()
        self.prompts = self.get_prompts()
        self.type = self.get_module_type()

    def __repr__(self):
        return self.name.replace("_", " ").title().replace(" ", "") + "()"

    def get_prompts(self):
        return ChatPromptTemplate.from_messages(MODULES_CONFIG[self.name]["prompts"].items())
    
    def get_parser(self):
        return PARSERS.get(MODULES_CONFIG[self.name]["parser"], StrOutputParser())
    
    def get_module_type(self):
        return MODULES_CONFIG[self.name]["type"]


class Pipeline:
    """Text2KG pipeline class."""

    def __init__(self, config: Config):
        
        self.config = config
        self.initialize(config)


    def __call__(self, text: str, clean: bool=True) -> pd.DataFrame:
        return self.run(text, clean)
    

    def __repr__(self) -> str:
        return f"Text2KG(\n\tconfig.pipeline={self.config.pipeline}\n\tconfig.llm={self.config.llm}\n)"
    

    def __str__(self) -> str:
        return ("[INPUT: text] -> " 
                + " -> ".join([str(m) for m in self.modules])
                + " -> [OUTPUT: knowledge graph]")

    
    @classmethod
    def from_ini(cls, config_path: str):
        return cls(Config.read_ini(config_path))
    
    
    def initialize(self, config: Config):
        """Initialize Text2KG pipeline from config."""
        
        # validate preprocess
        preprocess_modules = [Module(name) for name in config.pipeline["preprocess"]]
        
        for item in preprocess_modules:
            if item.get_module_type() != "preprocess":
                raise ValueError(f"Expected preprocess step `{item.name}` to"
                                 f" have module type='preprocess'. Consider reviewing"
                                 f" schema.yml")
        
        # validate extraction process
        extraction_module = Module(config.pipeline["extraction"])
        
        if extraction_module.get_module_type() != "extraction":
            raise ValueError(f"Expected `{extraction_module.name}` to have module"
                             f" type='extraction'. Consider reviewing schema.yml")

        # combine preprocess + extraction
        self.modules = preprocess_modules + [extraction_module]

        # init prompts & parsers
        prompts = [m.get_prompts() for m in self.modules]
        parsers = [m.get_parser() for m in self.modules]

        # init llm
        llm = ChatOpenAI(**self.config.llm)
        
        # init chains
        chains = [(prompt | llm | parser) 
                  for prompt, parser in zip(prompts, parsers)]

        # stitch chains together
        self.pipeline = {"text": RunnablePassthrough()} | chains[0]
        for i in range(1, len(chains)):
            self.pipeline = {"text": self.pipeline} | chains[i]
        
        # print pipeline
        print("Initialized Text2KG pipeline:")
        print(str(self))

    
    def run(self, text: str, clean=True) -> tuple[List[str], pd.DataFrame]:
        """Run Text2KG pipeline on passed text.
        
        Args:
            *texts (str): The text inputs
            clean (bool): Whether to clean the raw KG or not
        
        Returns:
            text_batches (list): Batched text
            knowledge_graph (DataFrame): A dataframe containing the extracted KG triplets, 
                indexed by batch
        """
        print("Running Text2KG pipeline:")
        # split text into batches
        text_batches = list(partition_sentences(
            sentences=sent_tokenize(text), 
            min_words=self.config.pipeline["words_per_batch"]
        ))

        # run pipeline in parallel; convert to dataframe
        print("Extracting knowledge graph...", end=' ')
        output = self.pipeline.batch(text_batches)
        
        knowledge_graph = pd.DataFrame([{'batch_id': i, **triplet} 
                                        for i, batch in enumerate(output) 
                                        for triplet in batch])
        
        if clean:
            knowledge_graph = self._clean(knowledge_graph)
        
        print("Done!", end='\n')
        
        return text_batches, knowledge_graph


    def _clean(self, kg: pd.DataFrame) -> pd.DataFrame:
        """Text2KG post-processing."""
        print("Cleaning knowledge graph components...", end=' ')
        drop_list = []

        for i, row in kg.iterrows():
            # drop stopwords (e.g. pronouns)
            if (row.subject in STOP_WORDS) or (row.object in STOP_WORDS):
                drop_list.append(i)

            # drop broken triplets
            elif row.hasnans:
                drop_list.append(i)
            
            # lowercase nodes/edges, drop articles
            else:
                article_pattern = r'^(the|a|an) (.+)'
                be_pattern = r'^(are|is) (a |an )?(.+)'

                kg.at[i, "subject"] = re.sub(article_pattern, r'\2', row.subject.lower())
                kg.at[i, "relation"] = re.sub(be_pattern, r'\3', row.relation.lower())
                kg.at[i, "object"] = re.sub(article_pattern, r'\2', row.object.lower()).strip('.')
        
        return kg.drop(drop_list)


    def _normalize(self):
        """Unused."""
        return
    
    def serialize(self):
        return self.config.serialize()