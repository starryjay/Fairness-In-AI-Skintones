import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def get_closest_images(representative_data: pd.DataFrame, morphe_data: pd.DataFrame, n: int=5) -> tuple[pd.DataFrame, list]:
    """
    Finds the closest Morphe swatches to the representative images based on the sum of the differences in R, G, and B values.
    Returns:
    - A DataFrame with the closest Morphe swatches and their distances.
    - A dictionary with the minimum summed distances for each representative image.
    """
    min_distances = {}
    for _, row in representative_data.iterrows():
        representative_rgb = np.array([row['Representative_R'], 
                                       row['Representative_G'], 
                                       row['Representative_B']])
        distances = {}
        for _, m_row in morphe_data.iterrows():
            morphe_rgb = np.array([m_row['R'], m_row['G'], m_row['B']])
            distance = np.sum(np.abs(representative_rgb - morphe_rgb))
            distances[(row['Image Path'], m_row['tone_name'])] = distance
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        n_closest = sorted_distances[:n]
        min_distances_row = defaultdict(dict)
        for img_tuple, distance in n_closest:
            img_path, morphe_img = img_tuple
            min_distances_row[img_path][morphe_img] = distance
        min_distances.update(min_distances_row)
    out = pd.DataFrame(columns=['Image Path', 'Morphe1', 'Dist_Morphe1', 
                                'Morphe2', 'Dist_Morphe2', 'Morphe3', 
                                'Dist_Morphe3', 'Morphe4', 'Dist_Morphe4', 
                                'Morphe5', 'Dist_Morphe5'])
    for img_path, morphe_dict in min_distances.items():
        row = [img_path]
        for i, (morphe_img, distance) in enumerate(morphe_dict.items()):
            row.append(morphe_img)
            row.append(distance)
        out.loc[len(out)] = row
    return out, min_distances

def write_csv(representative_data: pd.DataFrame, out: pd.DataFrame) -> None:
    """
    Writes the closest images and their distances to a CSV file.
    """
    if 'cropped_faces' in representative_data.iloc[0, 0]:
        out.to_csv('./intermediate_files/closest_images_fashion.csv', index=False)
    else:
        out.to_csv('./intermediate_files/closest_images_lfw.csv', index=False)

def plot_distances(min_distances: dict, rep_path: str) -> None:
    """
    Plots the distribution of the summed RGB distances from the nearest Morphe swatches.
    """
    distances_list = [list(morphe_dict.values()) for morphe_dict in min_distances.values()]
    min_distances = [distance for sublist in distances_list for distance in sublist]
    plt.figure(figsize=(10, 6), dpi=300)
    plt.hist(min_distances, bins=50, color='deepskyblue', edgecolor='black', linewidth=0.8)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    if 'fashion' in rep_path:
        plt.title('Distribution of Summed RGB Distances from Nearest Morphe Swatches - Fashion Dataset')
        plt.savefig('./plots/distance_distribution_fashion.png', dpi=300)
    else:
        plt.title('Distribution of Summed RGB Distances from Nearest Morphe Swatches - LFW Dataset')
        plt.savefig('./plots/distance_distribution_lfw.png', dpi=300)
    plt.show()

def main() -> None:
    """
    Main function to read the representative images and Morphe swatches, find the closest images, and plot the distances."""
    rep_path = './intermediate_files/representative_images_fashion.csv'
    #rep_path = './intermediate_files/representative_images_lfw.csv'
    morphe_data = pd.read_csv('./intermediate_files/morphe_swatch.csv')
    representative_data = pd.read_csv(rep_path)
    closest, min_distances = get_closest_images(representative_data, morphe_data, n=5)
    write_csv(representative_data, closest)
    plot_distances(min_distances, rep_path)

if __name__ == "__main__":
    main()