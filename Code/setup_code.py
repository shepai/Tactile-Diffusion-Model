import sys 
import os 
from pathlib import Path
current_dir = Path(__file__).resolve().parent
print("Current directory:", current_dir)
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import Data.datasetdownloader

 