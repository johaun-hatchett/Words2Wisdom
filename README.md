# AutoKG

Using large language models (ChatGPT) to automatically construct a knowledge graph from unstructured plain text.

## Usage

```
python text2kg.py <infile> [-h] [--output OUTPUT] [--cookbook COOKBOOK] [--recipe RECIPE] [--thoughts]
```

**Parameters**

```
infile
    path to input text file (assumed to be in plain text form)
--output
    directory to save results to (default: ./output)
--cookbook
    path to JSON file containing GPT prompts (default: ./recipes.json)
--recipe
    recipe to execute, must be a valid name from cookbook (see above)
--thoughts
    optional flag to save intermediary GPT prompts/replies
```

## File structure

### [`data`](./data/)

Name | Description | Source
--- | --- | ---
`data/openstax/...` | Sections from various OpenStax textbooks. | 
`data/Seq2KG/...` | Processed test datasets from Seq2KG paper. ("Processing" = merging tokens back to sentences) | [GitHub](https://github.com/Michael-Stewart-Webdev/Seq2KG/tree/master)

## References

1. A case study in bootstrapping ontology graphs from textbooks (V. K. Chaudhri et al., 2021)
2. Seq2KG: an end-to-end neural model for domain agnostic knowledge graph (not text graph) construction from text (M. Stewart & W. Liu, 2020)
3. Language models are open knowledge graphs (C. Wang et al., 2020)