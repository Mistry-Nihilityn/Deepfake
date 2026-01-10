import os.path

ROOT_PATH = "../dataset"
DATASET_NAMES = ["DiT", "lia", "heygen", "heygen_new", "deepfacelab", "StyleGAN2", "fsgan", "faceswap",
                 "SageMaker", "simswap", "stargan", "starganv2", "styleclip"]

if __name__ == "__main__":
    for dataset_name in DATASET_NAMES:
        if os.path.isfile(file:=os.path.join(ROOT_PATH, dataset_name, "label.json")):
            os.remove(file)
            print(f"removed {file} file")
