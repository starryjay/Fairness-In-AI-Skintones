import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def get_closest_images(representative_data: pd.DataFrame, morphe_data: pd.DataFrame, n: int=5) -> tuple[pd.DataFrame, list]:
    min_distances = {}
    for index, row in representative_data.iterrows():
        representative_rgb = np.array([row['Representative_R'], row['Representative_G'], row['Representative_B']])
        distances = {}
        for m_index, m_row in morphe_data.iterrows():
            morphe_rgb = np.array([m_row['R'], m_row['G'], m_row['B']])
            distance = np.sum(np.abs(representative_rgb - morphe_rgb))
            distances[(row['Image Path'], m_row['tone_name'])] = distance
        # get minimum distances
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        n_closest = sorted_distances[:n] # [((img_path, morphe_img), distance)]
        # convert to dict with structure {img_path: {morphe_img1: distance1, morphe_img2: distance2, ...}, ...}
        min_distances_row = defaultdict(dict)
        for img_tuple, distance in n_closest:
            img_path, morphe_img = img_tuple
            min_distances_row[img_path][morphe_img] = distance
        min_distances.update(min_distances_row)
    # nested dict to dataframe with columns ['Image Path', 'Morphe1', 'Dist_Morphe1', 'Morphe2', ..., 'Morphe5', 'Dist_Morphe5']
    out = pd.DataFrame(columns=['Image Path', 'Morphe1', 'Dist_Morphe1', 'Morphe2', 'Dist_Morphe2', 'Morphe3', 'Dist_Morphe3', 'Morphe4', 'Dist_Morphe4', 'Morphe5', 'Dist_Morphe5'])
    for img_path, morphe_dict in min_distances.items():
        row = [img_path]
        for i, (morphe_img, distance) in enumerate(morphe_dict.items()):
            row.append(morphe_img)
            row.append(distance)
        out.loc[len(out)] = row
    if 'cropped_faces' in representative_data.iloc[0, 0]:
        out.to_csv('closest_images_fashion.csv', index=False)
    else:
        out.to_csv('closest_images_lfw.csv', index=False)
    return out, min_distances

def plot_distances(min_distances: list) -> None:
    plt.hist(min_distances, bins=50)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances')
    plt.show()

def main() -> None:
    #rep_path = 'representative_images_fashion.csv'
    rep_path = 'representative_images_lfw.csv'
    morphe_data = pd.read_csv('morphe_swatch.csv')
    representative_data = pd.read_csv(rep_path)
    get_closest_images(representative_data, morphe_data, n=5)

if __name__ == "__main__":
    main()