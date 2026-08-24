import os

def extract_chunks():
    with open('c:/project-Graphnet/main.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # The line numbers are 1-indexed, Python lists are 0-indexed.
    # 708 means index 707.
    
    # Imports & Shared: 0 to 706
    # show_home_page: 707 to 903
    # show_data_collection_page: 904 to 1952
    # show_training_page: 1953 to 2946
    # show_evaluation_page: 2947 to 3386
    # show_detection_page: 3387 to 3967
    
    imports_and_shared = "".join(lines[0:707])
    
    # Add imports to each file
    base_imports = """
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
import time

from state_manager import *
from ui.utils import *
from model_registry import save_model_version, get_versions, load_model_version

# Additional specific imports...
"""

    os.makedirs('c:/project-Graphnet/ui/pages', exist_ok=True)
    
    # We will write the utils.py first
    with open('c:/project-Graphnet/ui/utils.py', 'w', encoding='utf-8') as f:
        f.write(imports_and_shared)
        
    # data_collection
    with open('c:/project-Graphnet/ui/pages/data_collection.py', 'w', encoding='utf-8') as f:
        f.write("from ui.utils import *\nfrom state_manager import *\n")
        f.write("".join(lines[904:1953]))

    # training
    with open('c:/project-Graphnet/ui/pages/training.py', 'w', encoding='utf-8') as f:
        f.write("from ui.utils import *\nfrom state_manager import *\n")
        f.write("".join(lines[1953:2947]))

    # evaluation
    with open('c:/project-Graphnet/ui/pages/evaluation.py', 'w', encoding='utf-8') as f:
        f.write("from ui.utils import *\nfrom state_manager import *\n")
        f.write("".join(lines[2947:3387]))

    # detection
    with open('c:/project-Graphnet/ui/pages/detection.py', 'w', encoding='utf-8') as f:
        f.write("from ui.utils import *\nfrom state_manager import *\n")
        f.write("".join(lines[3387:3968]))

if __name__ == '__main__':
    extract_chunks()
