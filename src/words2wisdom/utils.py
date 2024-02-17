import os
from datetime import datetime
from typing import List
from zipfile import ZipFile

import pandas as pd


def partition_sentences(sentences: List[str], min_words: int):
    current_batch = []
    word_count = 0
    
    for sentence in sentences:
        # count the number of words in the sentence
        word_count += len(sentence.split())
        
        # add sentence to the current batch
        current_batch.append(sentence) 
        
        # if the word count exceeds or equals the minimum threshold, yield the current batch
        if word_count >= min_words:
            yield " ".join(current_batch)
            current_batch = []  # reset the batch
            word_count = 0      # reset the word count
    
    # yield the remaining batch if it's not empty
    if current_batch:
        yield " ".join(current_batch)


def dump_all(pipeline, text_batches: List[str], knowledge_graph: pd.DataFrame, to_path: str=None):
    """Save all items to ZIP."""
    # metadata
    date = str(datetime.now().date())

    # convert batches to df
    batches_df = pd.DataFrame(text_batches, columns=["text"])

    # date + hex id for local saving
    num = 0
    while True:
        hex_num = format(num, 'X').zfill(3)
        filename = f"output-{date}-{hex_num}.zip"
        zip_path = os.path.join(to_path, filename)
        
        if os.path.exists(zip_path):
            num += 1
        else:
            break
    
    print(f"Run ID: {date}-{hex_num}")
    os.makedirs(to_path, exist_ok=True)
    
    # create ZIP file
    with ZipFile(zip_path, 'w') as zipObj:
        zipObj.writestr("config.ini", pipeline.serialize())
        zipObj.writestr("text_batches.csv", batches_df.to_csv(index_label="batch_id"))
        zipObj.writestr("kg.csv", knowledge_graph.to_csv(index=False))

    print(f"Saved data to {zip_path}")
    
    return zip_path