import gdown
import os
import shutil

def test():
    gdown.download("https://drive.google.com/uc?id=1WzXbErURkPyrXzHBoOrxs8bljrvNyMRD&export=download")

    os.makedirs("src/model_weights", exist_ok=True)
    shutil.move("model_best.pth", "src/model_weights/model_best.pth")


if __name__ == "__main__":
    test()
