import sys 
import os 
from pathlib import Path
import numpy as np
current_dir = Path(__file__).resolve().parent
print("Current directory:", current_dir)
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

if __name__=="__main__":
    import Data.datasetdownloader as dd
    from Data.convert_data import load_files_memory_efficient
    dd.download(target_path="/mnt/data0/drs25/data/nonlin")
    X,y=load_files_memory_efficient("/mnt/data0/drs25/data/nonlin/1")
    np.save(str(parent_dir)+"/Data/"+"X_Data",X)
    np.save(str(parent_dir)+"/Data/"+"y_Data",y)
    Xn,yn=load_files_memory_efficient("/mnt/data0/drs25/data/nonlin/2",type_="pressure")
    np.save(str(parent_dir)+"/Data/"+"Xn_Data",Xn)
    np.save(str(parent_dir)+"/Data/"+"yn_Data",yn)