import nltk

nltk.download("stopwords")

import re
from nltk.corpus import stopwords
from pandas import DataFrame

stop_words = stopwords.words("english")

def process(df: DataFrame):
    """Text2KG post-processing."""
    drop_list = []

    for i, row in df.iterrows():
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

            df.at[i, "subject"] = re.sub(article_pattern, r'\2', row.subject.lower())
            df.at[i, "relation"] = re.sub(be_pattern, r'\3', row.relation.lower())
            df.at[i, "object"] = re.sub(article_pattern, r'\2', row.object.lower())

    return df.drop(drop_list)