import configparser
import ast


class Config:
    def __init__(self, config_data):
        self.config_data = config_data

    def __getattr__(self, name):
        return self.config_data.get(name, {})

    def __setattr__(self, name, value):
        if name == 'config_data':
            super().__setattr__(name, value)
        else:
            self.config_data[name] = value

    def __repr__(self):
        return f"Config(\n{'pipeline':>12}: {self.pipeline}\n{'llm':>12}: {self.llm}\n)"

    @classmethod
    def read_ini(cls, file_path):
        
        parser = configparser.ConfigParser()
        parser.read(file_path)
        
        return cls({"pipeline":  cls._parse_pipeline_section(parser["pipeline"]),
                    "llm": cls._parse_llm_section(parser["llm"])})

    @staticmethod
    def _parse_llm_section(section):
        parsed_data = {}
        for key, value in section.items():
            try:
                parsed_data[key] = ast.literal_eval(value)
            except ValueError:
                parsed_data[key] = value

        return parsed_data
    
    @staticmethod
    def _parse_pipeline_section(section):
        eval_func = {
            "words_per_batch": int,
            "preprocess": lambda x: x.split(", ") if x.split(", ") != ["None"] else []
        }
        parsed_data = {}

        for key, value in section.items():
            parsed_data[key] = eval_func.get(key, str)(value)

        return parsed_data
    

    def serialize(self, save_path: str=None):
        """Convert Config object to .ini file. If save_path is not specified, return string"""
        serialized_config = ''
        
        for section in self.config_data:
            serialized_config += f"[{section}]\n"
            
            for key, value in self.config_data[section].items():
                # turn list back to str
                if isinstance(value, list):
                    value = ", ".join(value)
                
                # don't serialize the api key
                if key == "openai_api_key":
                    value = None

                serialized_config += f"{key} = {value}\n"
            
            serialized_config += "\n"

        if save_path:
            with open(save_path, 'w') as f:
                f.write(serialized_config)
        else:
            return serialized_config 



if __name__ == "__main__":
    # example usage
    config_file = "/Users/johaunh/Documents/PhD/Projects/Text2KG/config/config.ini"
    config = Config.read_ini(config_file)

    # access pipeline parameters
    print("Pipeline Parameters:")
    for k, v in config.pipeline.items():
        print(f"{k}: {v}")

    # access LLM parameters
    print("\nLLM Parameters:")
    for k, v in config.llm.items():
        print(f"{k}: {v}")
