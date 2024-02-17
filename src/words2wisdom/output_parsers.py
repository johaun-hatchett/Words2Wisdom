import re

import pandas as pd
from langchain_core.output_parsers import NumberedListOutputParser, StrOutputParser


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
    

class QuestionOutputParser(StrOutputParser):
    def get_format_instructions(self) -> str:
        return super().get_format_instructions()
    
    def parse(self, text: str) -> pd.DataFrame:
        """Parses the response to an array of answers/explanations."""
        output = super().parse(text)
        raw_list = re.findall(r'\d+\) (.*?)(?=\n\d+\)|\Z)', output, re.DOTALL)
        
        raw_df = pd.DataFrame(raw_list).T
        
        df = pd.DataFrame()

        for idx in raw_df.columns:
            # answer and explanation headers
            ans_i = f"Q{idx+1}"
            why_i = f"Q{idx+1}_explain"
            
            # split response into answer/explanation columns
            df[[ans_i, why_i]] = raw_df[idx].str.extract(r'(\d) \- (.*)')

        return df