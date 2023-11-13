---
title: Text2KG
app_file: main.py
sdk: gradio
sdk_version: 3.39.0
pinned: true
---
# Text2KG

Using large language models (ChatGPT) to automatically construct a knowledge graph from unstructured plain text.

## Usage

### Local Gradio app

To locally launch the `Gradio` app from the command line, use

```
python main.py
```

### Within a `python` IDE

Import the primary pipeline method using

```python
>>> from main import create_knowledge_graph
```

**`create_knowledge_graph` parameters**

```
api_key (str)
    OpenAI API key
ngram_size (int)
    Number of sentences per forward pass
axiomatize (bool)
    Whether to decompose sentences into simpler axioms as a
    pre-processing step. Doubles the amount of calls to ChatGPT
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
    Converts schema.yml items to LangChain chains

environment.yml
    Contains packages required to run environment

main.py
    Main pipeline/app code

README.md
    This file

schema.yml
    Contains definitions of prompts

utils.py
    Contains helper functions
```