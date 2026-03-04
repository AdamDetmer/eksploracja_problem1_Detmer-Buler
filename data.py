import kagglehub
import shutil
import os

downloaded_path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")

current_dir = os.path.dirname(os.path.abspath(__file__))
target_data_dir = os.path.join(current_dir, "data")

os.makedirs(target_data_dir, exist_ok=True)

for item in os.listdir(downloaded_path):
    s = os.path.join(downloaded_path, item)
    d = os.path.join(target_data_dir, item)

    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print(f"Sukces! Dane zostały skopiowane do: {target_data_dir}")