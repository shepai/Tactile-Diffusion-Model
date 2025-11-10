import sys 
import os 
current_dir = os.getcwd()
print("Current directory:", current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(str(parent_dir))

import Data.datasetdownloader
