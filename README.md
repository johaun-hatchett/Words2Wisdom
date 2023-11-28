---
title: Text2KG
app_file: main.py
sdk: gradio
sdk_version: 3.39.0
pinned: true
license: mit
emoji: 🧞📖
colorFrom: indigo
colorTo: gray
---
# Text2KG

We introduce Text2KG – an intuitive, domain-independent tool that leverages the creative generative ability of GPT-3.5 in the KG construction process. Text2KG automates and accelerates the construction of KGs from unstructured plain text, reducing the need for traditionally-used human labor and computer resources. Our approach incorporates a novel, clause-based text simplification step, reducing the processing of even the most extensive corpora down to the order of minutes. With Text2KG, we aim to streamline the creation of databases from natural language, offering a robust, cost-effective, and user-friendly solution for KG construction.

## Usage

### Gradio app

#### Remotely

Visit the [Text2KG HuggingFace Space](https://huggingface.co/spaces/jhatchett/Text2KG).

#### Locally

Clone this repository, and then use the command

```
python main.py
```

in the repository's directory.

### Within a `python` IDE

Import the primary pipeline method using

```python
>>> from main import extract_knowledge_graph
```

**`extract_knowledge_graph` parameters**

```
api_key (str)
    OpenAI API key

batch_size (int)
    Number of sentences per forward pass

modules (list)
    Additional modules to add before main extraction process (triplet_extraction). Must be a valid name in schema.yml

text (str)
    Input text to extract knowledge graph from

progress
    Progress bar. The default is Gradio's progress bar; 
    set `progress = tqdm` for implementations outside of Gradio
```

### Using Gradio API

Read more [here](https://www.gradio.app/docs/python-client).

## File structure

```
chains.py
    Converts the items in schema.yml to LangChain modules

requirements.txt
    Contains packages required to run Text2KG

main.py
    Main pipeline/app code

README.md
    This file

schema.yml
    Contains definitions of modules -- prompts + output parsers

utils.py
    Contains helper functions
```