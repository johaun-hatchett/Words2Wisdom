import argparse
import os
import subprocess

from langchain_openai import ChatOpenAI

from . import CONFIG_DIR, OUTPUT_DIR
from .pipeline import Pipeline
from .utils import dump_all
from .validate import validate_knowledge_graph


default_config_path = os.path.join(CONFIG_DIR, "default_config.ini")


def main():
    parser = argparse.ArgumentParser(
        prog="words2wisdom",
        description="Knowledge graph generation utilities using OpenAI LLMs"
    )
    subparsers = parser.add_subparsers(dest="command", 
                                       help="Available commands")

    # init
    parser_init = subparsers.add_parser("init", 
                                        usage="words2wisdom init [> PATH/TO/WRITE/CONFIG.INI]",
                                        help="Initialize a template config.ini file",
                                        description="Initialize a template config.ini file. Redirect to a new file using the '>' symbol.")
    parser_init.set_defaults(func=get_default_config)

    # gui
    parser_gui = subparsers.add_parser("gui", 
                                       help="Use Words2Wisdom via Gradio interface",
                                       description="use Words2Wisdom using Gradio interface")
    parser_gui.add_argument("-s", "--streamlit",
                            action="store_true",
                            help="Use Streamlit GUI instead of Gradio GUI")
    parser_gui.set_defaults(func=gui)

    # run
    parser_run = subparsers.add_parser("run",
                                       help="Generate a knowledge graph from a given text using OpenAI LLMs",
                                       description="Generate a knowledge graph from a given text using OpenAI LLMs")
    parser_run.add_argument("text", 
                            help="Path to text file")
    parser_run.add_argument("-c", "--config",
                            help="Path to config.ini file",
                            default=default_config_path)
    parser_run.add_argument("-o", "--output-dir", 
                            metavar="OUTPUT_PATH",
                            help="Path to save outputs to", 
                            default=OUTPUT_DIR)
    parser_run.set_defaults(func=run)

    # eval
    parser_eval = subparsers.add_parser("eval", 
                                        help="Auto-evaluate knowledge graph using GPT-4",
                                        description="Auto-evaluate knowledge graph using GPT-4")
    parser_eval.add_argument("output_zip", 
                             help="Path to output.zip file containing knowledge graph")
    parser_eval.set_defaults(func=validate)

    args = parser.parse_args()
    args.func(args)


def get_default_config(args):
    """Print default config.ini"""
    with open(default_config_path) as f:
        default_config = f.read()
    
    print(default_config)


def gui(args):
    """Run interface"""
    if args.streamlit:
        cmd = "streamlit run words2wisdom/gui_streamlit.py".split()
    else:
        cmd = "python -m words2wisdom.gui".split()

    subprocess.run(cmd)


def run(args):
    """Text to KG pipeline"""
    pipe = Pipeline.from_ini(args.config)

    with open(args.text) as f:
        batches, kg = pipe.run(f.read())
    
    dump_all(pipe, batches, kg, to_path=args.output_dir)

    
def validate(args):
    """Validate knowledge graph"""
    validate_knowledge_graph(
        llm=ChatOpenAI(
            model="gpt-4-turbo-preview",
            #openai_api_key=...
        ), 
        output_zip=args.output_zip
    )


if __name__ == "__main__":
    main()
