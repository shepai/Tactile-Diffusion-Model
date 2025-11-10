import kagglehub
import subprocess

def download(target_path="/mnt/data0/drs25/data/nonlin"):
    pathA = kagglehub.dataset_download("dextershepherd/tactip-nonlinear-alternative-morphology-b")
    pathB = kagglehub.dataset_download("dextershepherd/tactip-alternative-morphology-b")
    print("Path to dataset files:", pathA)
    print("Path to dataset files:", pathB)
    result = subprocess.run(["mv", pathA, target_path,"/1"], capture_output=True, text=True)
    result = subprocess.run(["mv", pathB, target_path,"/2"], capture_output=True, text=True)
    subprocess.run(["rm",pathA])
    subprocess.run(["rm",pathB])
