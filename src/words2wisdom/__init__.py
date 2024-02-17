import os
import sys
import yaml

import nltk

# directories
PACKAGE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(PACKAGE_DIR))
DATA_DIR = os.path.join(ROOT, "data")
CONFIG_DIR = os.path.join(ROOT, "config")
OUTPUT_DIR = os.path.join(ROOT, "output")

# add the package root directory to the python path
sys.path.append(os.path.dirname(PACKAGE_DIR))

# files
with open(os.path.join(CONFIG_DIR, "modules.yml")) as f:
    MODULES_CONFIG = yaml.safe_load(f)

with open(os.path.join(CONFIG_DIR, "validation.yml")) as f:
    VALIDATION_CONFIG = yaml.safe_load(f)

# download NLTK dependencies
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# load NLTK stop words
from nltk.corpus import stopwords
STOP_WORDS = stopwords.words("english")