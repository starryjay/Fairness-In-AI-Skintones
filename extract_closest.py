import pandas as pd
import os

def extract_closest_images(paths):
    # if the directory exists, delete it and its contents
    if os.path.exists('closest_images'):
        print("Deleting closest_images directory")
        os.system('rm -rf closest_images')
    os.makedirs('closest_images', exist_ok=True)
    for index, img in paths.iterrows():
        img = img.iloc[0]
        img_path = os.path.join('closest_images', img.split('/')[-1])
        print(img_path)
        if not os.path.exists(img_path):
            print(f"Copying {img} to closest_images")
            os.system(f'cp {img} {img_path}')
        else:
            print(f"{img} already exists in closest_images")

if __name__ == "__main__":
    img_paths = pd.read_csv('closest_images.csv').loc[:, ['Representative_Image']]
    extract_closest_images(img_paths)