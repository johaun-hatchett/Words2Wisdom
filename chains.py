from functools import partial

import yaml
from langchain.chains import LLMChain
from langchain.output_parsers import NumberedListOutputParser
from langchain.prompts import ChatPromptTemplate


with open("./schema.yml") as f:
    schema = yaml.safe_load(f)


class ClauseParser(NumberedListOutputParser):

    def parse(self, text: str) -> str:
        axioms = super().parse(text=text)
        return " ".join(axioms)

    def get_format_instructions(self) -> str:
        return super().get_format_instructions()
    

class TripletParser(NumberedListOutputParser):
    
    def parse(self, text: str) -> str:

        output = super().parse(text=text)
        headers = ["subject", "relation", "object"]
        triplets = [dict(zip(headers, item.split("::"))) for item in output]

        return triplets

    def get_format_instructions(self) -> str:
        return super().get_format_instructions()


llm_chains = {}

for scheme in schema:
    parser = schema[scheme]["parser"]
    prompts = schema[scheme]["prompts"]

    llm_chains[scheme] = partial(
        LLMChain, 
        output_parser=eval(f'{parser}()'), 
        prompt=ChatPromptTemplate.from_messages(list(prompts.items()))
    )