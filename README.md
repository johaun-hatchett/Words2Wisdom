# Words2Wisdom

This is the repository for Words2Wisdom. The work is still a work in progress.

**Paper:** [here](./writeup/words2wisdom_short.pdf) (Accepted as poster to AAAI AI4ED '24 Workshop)

**Hugging Face Space:** [Words2Wisdom](https://huggingface.co/spaces/jhatchett/Words2Wisdom)

**Abstract:**
Large language models (LLMs) have emerged as powerful tools with vast potential across various domains. While they have the potential to transform the educational landscape with personalized learning experiences, these models face challenges such as high training and usage costs, and susceptibility to inaccuracies. One promising solution to these challenges lies in leveraging knowledge graphs (KGs) for knowledge injection. By integrating factual content into pre-trained LLMs, KGs can reduce the costs associated with domain alignment, mitigate the risk of hallucination, and enhance the interpretability of the models' outputs. To meet the need for efficient knowledge graph creation, we introduce *Words2Wisdom* (W2W), a domain-independent LLM-based tool that automatically generates KGs from plain text. With W2W, we aim to provide a streamlined KG construction option that can drive advancements in grounded LLM-based educational technologies.

## Demonstration

The `demo/demo.ipynb` notebook walks through how to use the `words2wisdom` pipeline.

## Usage

Due to the large number of configurable parameters, `words2wisdom` uses a configuration INI file:

```ini
[pipeline]
words_per_batch = 150 # any positive integer
preprocess = clause_deconstruction # {None, clause_deconstruction}
extraction = triplet_extraction # {triplet_extraction}

[llm]
model = gpt-3.5-turbo
# other GPT params like temperature, etc. can be set here too
```

A template configuration file can be generated with the command-line interface. **Note:** If `openai_api_key` is not explicitly set, the config will automatically try to read from the `OPENAI_API_KEY` environment variable.

### From the CLI

All commands are preceded by `python -m words2wisdom`

| In order to... | Use the command... |
| -- | -- |
| Create a new config file | `init > path/to/write/config.ini` |
| Generate KG from text | `run path/to/text.txt [--config CONFIG] [--output-dir OUTPUT_DIR]` |
| Evaluate `words2wisdom` outputs | `eval path/to/output.zip` |
| Use `words2wisdom` from Gradio interface (default config only) | `gui` |

### As a `Python` package

Import the primary pipeline method using

```python
from words2wisdom.pipeline import Pipeline

# configure pipeline from .ini
pipe = Pipeline.from_ini("path/to/config.ini")
text_batches, knowledge_graph = pipe.run("The cat sat on the mat")
```

## File structure

```
├── config
│   ├── default_config.ini
│   ├── modules.yml
│   └── validation.yml
├── demo
│   ├── config.ini
│   ├── demo.ipynb
│   └── example.txt
├── src/words2wisdom
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── gui.py
│   ├── output_parsers.py
│   ├── pipeline.py
│   ├── utils.py
│   └── validate.py
├── writeup
│   └── words2wisdom_short.pdf
├── LICENSE.md
├── README.md
└── requirements.txt
```
