import os

import gradio as gr

from . import CONFIG_DIR, ROOT
from .config import Config
from .pipeline import Pipeline
from .utils import dump_all


example_file = (os.path.join(ROOT, "demo", "prokaryotes.txt"))
example_text = "The quick brown fox jumps over the lazy dog. The cat sits on the mat."


def text2kg_from_string(openai_api_key: str, input_text: str):

    config = Config.read_ini(os.path.join(CONFIG_DIR, "default_config.ini"))
    config.llm["openai_api_key"] = openai_api_key

    pipeline = Pipeline(config)
    text_batches, knowledge_graph = pipeline.run(input_text)

    zip_path = dump_all(config, text_batches, knowledge_graph)

    return knowledge_graph, zip_path


def text2kg_from_file(openai_api_key: str, input_file):
    with open(input_file.name) as f:
        input_text = f.read()

    return text2kg_from_string(openai_api_key, input_text)


with gr.Blocks(title="Text2KG") as demo:
    gr.Markdown("# 🧞📖 Text2KG")
    
    with gr.Column(variant="panel"):
        openai_api_key = gr.Textbox(label="OpenAI API Key", placeholder="sk-...", type="password")
    
    with gr.Row(equal_height=False):
        with gr.Column(variant="panel"):
            gr.Markdown("## Input (Text or Text File)")
            #gr.Markdown("A knowledge graph will be generated for the provided text.")
            with gr.Tab("Direct Input"):
                text_string = gr.Textbox(lines=2, placeholder="Text Here...", label="Text")
                submit_str = gr.Button()

            with gr.Tab("File Upload"):
                text_file = gr.File(file_types=["text"], label="Text File")
                submit_file = gr.Button()
        
        with gr.Column(variant="panel"):
            gr.Markdown("## Output (ZIP Archive)")
            #gr.Markdown("The ZIP contains the generated knowledge graph, the text batches (indexed), and a configuration file.")
            output_zip = gr.File(label="ZIP")
    
    with gr.Accordion(label="Preview of Knowledge Graph", open=False):
        output_graph = gr.DataFrame(headers=["batch_id", "subject", "relation", "object"], label="Knowledge Graph")

    with gr.Accordion(label="Examples", open=False):
        gr.Markdown("### Text Example")
        gr.Examples(
            examples=[[None, example_text]],
            inputs=[openai_api_key, text_string],
            outputs=[output_graph, output_zip],
            fn=text2kg_from_string,
            preprocess=False,
            postprocess=False
        )
        
        gr.Markdown("### File Example")
        gr.Examples(
            examples=[[None, example_file]],
            inputs=[openai_api_key, text_file],
            outputs=[output_graph, output_zip],
            fn=text2kg_from_file,
            preprocess=False,
            postprocess=False
        )

    submit_str.click(fn=text2kg_from_string, inputs=[openai_api_key, text_string], outputs=[output_graph, output_zip])
    submit_file.click(fn=text2kg_from_file, inputs=[openai_api_key, text_file], outputs=[output_graph, output_zip])


demo.launch(inbrowser=True, width="75%")