import pandas as pd
import os

def extract_closest_images(paths: pd.Series, append: str) -> None:
    """
    Extracts the closest Morphe swatches to the given representative images and saves them to a new directory.
    """
    if os.path.exists('closest_images' + append):
        print("Deleting closest_images directory")
        os.system('rm -rf closest_images' + append)
    os.makedirs('closest_images' + append, exist_ok=True)
    for index, img in paths.iterrows():
        img = img.iloc[0]
        img_path = os.path.join('closest_images' + append, img.split('/')[-1])
        print(img_path)
        if not os.path.exists(img_path):
            os.system(f'cp {img} {img_path}')
            print(f"Copied {img} to closest_images" + append)
        else:
            print(f"{img} already exists in closest_images" + append)

def main() -> None:
    """
    Main function to read image paths and extract closest images.
    """
    #append = '_fashion'
    #append = '_lfw'
    append = '_overall'
    img_paths = pd.read_csv('./intermediate_files/closest_images' + append + '.csv').loc[:, ['Image Path']]
    extract_closest_images(img_paths, append)

if __name__ == "__main__":
    main()