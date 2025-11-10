import kagglehub
import subprocess

pathA = kagglehub.dataset_download("dextershepherd/tactip-nonlinear-alternative-morphology-b")
pathB = kagglehub.dataset_download("dextershepherd/tactip-alternative-morphology-b")
print("Path to dataset files:", pathA)
print("Path to dataset files:", pathB)
#result = subprocess.run(["mv", pathA, "/mnt/data0/drs25/data/"], capture_output=True, text=True)
