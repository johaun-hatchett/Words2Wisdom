import os

import openai
from colorama import Fore
from tenacity import retry, wait_random_exponential, stop_after_attempt


openai.organization = os.getenv("OPENAI_ORG_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatGPT:
    """
    Wrapper for making API calls to ChatGPT.

    Initializing the API calls:
        >>> openai.api_key = ...
        >>> chat = ChatGPT(model="gpt-3.5-turbo")
    Usage:
        >>> chat("Hello ChatGPT!")
        Hello! How can I assist you today?
    """
    
    def __init__(self, model: str, init: str = None, history: list = None, **options) -> None:
        """
        Args:
            model (str): name of OpenAI chat completion model to use
            init (str): system initialization command (default: None)
            history (list): list of chat exchanges to preload (default: None)
        
        Optional Kwargs:
            temperature, ... (see OpenAI docs for more information)
        """
        self.model = model
        self.options = options
        self.init = init
        
        if history is None:
            self.history = []
        else:
            self.history = history.copy()
        
        if init is not None:
            self.history.append({"role": "system", "content": init})

    
    @retry(
        wait=wait_random_exponential(multiplier=1, max=40), # exponential backoff
        stop=stop_after_attempt(3) # try 3x before giving up
    )
    def __call__(self, prompt: str) -> str | None:
        """Attempt to make a call to chatGPT with the specified prompt."""
        self.history.append({"role": "user", "content": prompt})

        try:
            response = openai.ChatCompletion.create(model=self.model, messages=self.history, **self.options)
            reply = response.choices[0].message.content

            self.history.append({"role": "assistant", "content": reply})
            self.last_response = response
        
        except Exception as e:
            reply = (
                f"Unable to generate ChatCompletion response.\n"
                f"Exception: {e}"
            )

            print(Fore.LIGHTRED_EX + reply + Fore.RESET)
        
        finally:
            return reply
        
    
    def __repr__(self) -> str:
        prefix = f"ChatGPT(model={self.model},"
        infix = ", ".join(
            [f"{setting}={value}" for setting, value in self.options.items()]
        )
        suffix = ")"
        return prefix + infix + suffix

    
    def clear_history(self) -> None:
        """Clear chat history."""
        self.history = []
        
        if self.init is not None:
            self.history.append({"role": "system", "content": self.init})