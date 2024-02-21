import io
import os
import sys
from zipfile import ZipFile

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components
from pandas import DataFrame
from pyvis.network import Network

from words2wisdom import CONFIG_DIR
from words2wisdom.config import Config as W2WConfig
from words2wisdom.pipeline import Pipeline


def create_graph(df: DataFrame):
    graph = Network(directed=True)

    entities = pd.concat([df.subject, df.object]).unique()

    graph.add_nodes(entities, label=entities, title=entities)

    df_iterable = (
        df.drop_duplicates(
            subset=["subject", "relation", "object"]
        )
        .iterrows()
    )

    for _, row in df_iterable:
        graph.add_edge(row.subject, row.object, label=row.relation)

    graph.save_graph("/tmp/graph.html")
    HtmlFile = open("/tmp/graph.html")
    
    return st_components.html(HtmlFile.read(), height=625)


@st.cache_data
def create_zip_bytes(file_contents):
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w') as zip_file:
        for filename, content in file_contents.items():
            zip_file.writestr(filename, content)
    return buffer.getvalue()


st.set_page_config(page_title="Words2Wisdom",
                   page_icon="📖")
st.title("📖 Words2Wisdom")
st.write("Generate knowledge graphs from unstructured text using GPT.")

# parameters
with st.sidebar:
    st.title("Parameters")

    st.write("The API Key is required. Feel free to customize the other parameters, if you'd like!")

    openai_api_key = st.text_input(
        label="🔐 **OpenAI API Key**", 
        type="password",
        help="Learn how to get your own [here](https://platform.openai.com/docs/api-reference/authentication)."
    )
    st.divider()

    with st.expander("🚰 **Pipeline parameters**"):

        formatter = lambda x: x.replace("_", " ").title()

        words_per_batch = st.number_input(
            label="Words per Batch", 
            min_value=0, 
            max_value=200,
            value=150,
            help="Batch text into paragraphs containing at least N words, if possible."
        )

        preprocess = st.selectbox(
            label="Preprocess", 
            options=("None", "clause_deconstruction"), 
            index=1,
            format_func=formatter, 
            help="Method for text simplification."
        )

        extraction = st.selectbox(
            label="Generation", 
            options=("triplet_extraction",), 
            index=0,
            format_func=formatter,
            help="Method for KG generation."
        )

    with st.expander("🤖 **LLM parameters**"):
        model = st.selectbox(
            label="Model",
            options=("gpt-3.5-turbo",),
            index=0,
            help="ID of the model to use."
        )

        temperature = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            format="%.1f",
            help=(
                "What sampling temperature to use."
                " Higher values will make the output more random;"
                " lower values will make it more focused/deterministic."
            )
        )


# API Key warning 
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key.', icon='⚠️')


# tabs
tab1, tab2 = st.tabs(["Input Text", "File Upload"])


# text input tab
with tab1:
    text1 = st.text_area(label="Enter text:")
    submitted1 = tab1.button(label="Generate!", use_container_width=True)


# file upload tab
with tab2:
    file2 = tab2.file_uploader(label="Upload text file:", type="txt")
    submitted2 = tab2.button(key="filebtn", label="Generate!", use_container_width=True)


# w2w config
w2w_config = W2WConfig.read_ini(os.path.join(CONFIG_DIR, "default_config.ini"))
w2w_config.pipeline = {
    "words_per_batch": words_per_batch,
    "preprocess": [] if preprocess == "None" else [preprocess],
    "extraction": extraction
}
w2w_config.llm["openai_api_key"] = openai_api_key


# main logic
if (submitted1 or submitted2) and openai_api_key.startswith("sk-"):
    with st.status("Generating knowledge graph..."):
        st.write("Initializing pipeline...")
        pipe = Pipeline(w2w_config)
        st.write("Executing pipeline...")

        if submitted1:
            text = text1
        elif submitted2:
            text = file2.read().decode()
        
        text_batches, knowledge_graph = pipe.run(text)
        st.write("Complete.")

    st.divider()
    
    kg_viz = create_graph(knowledge_graph)

    st.error("**Warning:** The page will refresh when you download the data!", icon="🚨")

    download = st.download_button(
        label="Download data",
        data=create_zip_bytes({
            "text_batches.csv": (
                DataFrame(text_batches, columns=["text"])
                .to_csv(index_label="batch_id")
            ),
            "kg.csv": knowledge_graph.to_csv(index=False),
            "config.ini": pipe.serialize()
        }),
        file_name="output.zip",
        use_container_width=True,
        type="primary"
    )

