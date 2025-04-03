import cv2
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def extract_skin(image: np.ndarray) -> tuple[int, int]:
    """
    Extracts the skin tone from a 20px x 20px square in the center of the image.
    Returns: 
    - A tuple of the average RGB values of the skin tone.
    """
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2
    # get pixel value of area around center pixel
    center_pixel = image[center_y-10:center_y+10, center_x-10:center_x+10]
    # calculate average RGB values
    avg_rgb = np.mean(center_pixel, axis=(0, 1))
    return avg_rgb

def get_image_paths(root_dir: str) -> list[str]:
    """
    Get a list of image paths from the specified directory.
    Returns:
    - A list of image paths.
    """
    print('getting image paths')
    image_paths = []
    for person_folder in os.listdir(root_dir):
        person_path = os.path.join(root_dir, person_folder)
        if os.path.isdir(person_path):
            img_path = os.path.join(person_path, np.random.choice(os.listdir(person_path)))
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(img_path)
    image_paths = np.random.choice(image_paths, size=int(len(image_paths) * 0.8), replace=False)
    print('got image paths')
    return image_paths

def process_images(image_paths: list[str]) -> dict[str, dict[str, float]]:
    """
    Process images to extract skin tones.
    Returns:
    - A nested dictionary with image paths as keys and RGB dicts as values.
    """
    print('processing images')
    skin_tones = {}
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        skin = extract_skin(img)
        skin_tones[path] = {'R': float(skin[0]), 'G': float(skin[1]), 'B': float(skin[2])}
    print('processed images')
    return skin_tones

def cluster_images(skin_tones: dict[str, dict[str, float]], image_paths: list[str], n_clusters: int=10) -> tuple[np.ndarray, np.ndarray, dict[int, list[str]]]:
    """
    Cluster images based on skin tones using KMeans clustering.
    Returns:
    - A tuple containing the labels, cluster centers, representative images, and number of clusters.
    """
    print('clustering images')
    tones = []
    for d in skin_tones.values():
        tones.append((d['R'], d['G'], d['B']))
    tones = np.array(tones)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    labels = kmeans.fit_predict(tones)
    cluster_centers = kmeans.cluster_centers_
    representative_images = {}
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_skin_tones = tones[cluster_indices]
        distances = np.linalg.norm(cluster_skin_tones - cluster_centers[i], axis=1)
        closest_indices = cluster_indices[np.argsort(distances)[:4]]
        representative_images[i] = [image_paths[idx] for idx in closest_indices]
    print('clustered images')
    return (labels, cluster_centers, representative_images, n_clusters)

def plot_cluster_sizes(labels: np.ndarray, cluster_centers: np.ndarray, image_paths: list[str], n_clusters: int=10) -> None:
    """
    Plot the sizes of each cluster and save the plot.
    """
    # plot size of each cluster, colored by cluster center rgb
    plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.subplot(111)
    _, _, patches = ax.hist(labels, bins=n_clusters, edgecolor='black', linewidth=1)
    plt.bar_label(patches)
    for i in range(n_clusters):
        color = cluster_centers[i] / 255
        patches[i].set_facecolor(color)
        patches[i].set_edgecolor('black')
        patches[i].set_linewidth(1)
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.xticks(range(n_clusters))
    plt.grid(axis='y')
    if 'cropped_faces' in image_paths[0]:
        plt.title('Cluster Sizes for Fashion Dataset')
        plt.savefig('./plots/cluster_sizes_fashion.png')
    else:
        plt.title('Cluster Sizes for LFW Dataset')
        plt.savefig('./plots/cluster_sizes_lfw.png')
    plt.show()

def process_img_for_clustering(df: pd.DataFrame, representatives: dict[int, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process images for clustering by extracting skin tones and merging with the original dataframe.
    Returns:
    - A tuple containing the merged dataframe and a dataframe with skin tones.
    """
    print('processing images for clustering')
    skin_tones = {}
    all_paths = [img_path for img_list in representatives.values() for img_path in img_list]
    all_skin_tones = process_images(all_paths)
    for cluster, img_list in representatives.items():
        skin_tones[cluster] = [(img_path, all_skin_tones[img_path]) for img_path in img_list if img_path in all_skin_tones]
    skintone_df = pd.DataFrame.from_dict(skin_tones, orient='index')
    skintone_df = skintone_df.stack().reset_index()
    skintone_df.columns = ['Cluster', 'Representative', 'RGB']
    # scale cluster from 1-10
    skintone_df['Cluster'] = skintone_df['Cluster'].astype(int)
    skintone_df[['Image Path', 'RGB']] = pd.DataFrame(skintone_df['RGB'].tolist(), index=skintone_df.index)
    skintone_df[['R', 'G', 'B']] = pd.DataFrame(skintone_df['RGB'].tolist(), index=skintone_df.index)
    skintone_df = skintone_df.drop(columns=['RGB'])
    skintone_df.loc[:, 'R'] = skintone_df.loc[:, 'R'].astype(int) / 255
    skintone_df.loc[:, 'G'] = skintone_df.loc[:, 'G'].astype(int) / 255
    skintone_df.loc[:, 'B'] = skintone_df.loc[:, 'B'].astype(int) / 255
    skintone_df.loc[:, 'RGB_tuple'] = skintone_df.loc[:, ['R', 'G', 'B']].apply(tuple, axis=1)
    df.loc[:, 'Cluster'] = df.index
    df.loc[:, 'R'] = df.loc[:, 'R'] / 255
    df.loc[:, 'G'] = df.loc[:, 'G'] / 255
    df.loc[:, 'B'] = df.loc[:, 'B'] / 255
    df.loc[:, 'RGB_tuple'] = df.loc[:, ['R', 'G', 'B']].apply(tuple, axis=1)
    df = pd.merge(df, skintone_df, on='Cluster', how='outer').rename(columns={'R_x': 'Cluster_R', 'G_x': 'Cluster_G', 'B_x': 'Cluster_B', 
                                                                             'R_y': 'Representative_R', 'G_y': 'Representative_G', 'B_y': 'Representative_B', 
                                                                             'RGB_tuple_x': 'Cluster_RGB_tuple', 'RGB_tuple_y': 'Representative_RGB_tuple'})
    print('processed images for clustering')
    return df, skintone_df

def print_skintone_distributions(skintone_df: pd.DataFrame, labels: np.ndarray) -> None:
    print('first 10 items in labels:\n', labels[:10])
    #cluster_dist = pd.crosstab(index=)
    

def plot_clusters(df: pd.DataFrame) -> None:
    """
    Plot the clusters in 3D space with R, G, and B as axes and save the plot.
    """
    print('plotting clusters')
    fig = plt.figure(figsize=(18, 18), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    markers = ['o', 's', 'd', '^', '8', 'P', 'X', 'p', '*', 'h']
    unique_clusters = df['Cluster'].unique()
    cluster_marker_map = {cluster: markers[i % len(markers)] for i, cluster in enumerate(unique_clusters)}
    cluster_color_map = df.loc[:, ['Cluster', 'Cluster_RGB_tuple']].set_index('Cluster').to_dict()['Cluster_RGB_tuple']
    for cluster in unique_clusters:
        cluster_data = df[df['Cluster'] == cluster]
        ax.scatter(cluster_data['Cluster_R'], cluster_data['Cluster_G'], cluster_data['Cluster_B'], 
                   c=[cluster_color_map[cluster]], 
                   s=500, 
                   marker=cluster_marker_map[cluster], 
                   edgecolors='black',
                   linewidth=0.8,
                   label=f'Cluster {cluster + 1}')
    for cluster in unique_clusters:
        rep_data = df[df['Cluster'] == cluster]
        ax.scatter(rep_data['Representative_R'], rep_data['Representative_G'], rep_data['Representative_B'], 
                   c=rep_data[['Representative_R', 'Representative_G', 'Representative_B']].values / 255, 
                   s=200,
                   marker=cluster_marker_map[cluster]
                   )
    for i, row in df.iterrows():
        ax.text(row['Cluster_R'], row['Cluster_G'], row['Cluster_B'], str(row['Cluster'] + 1), size=20, zorder=100)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_xticklabels((np.arange(120, 251, 20)), fontsize=15)
    ax.set_yticklabels((np.arange(60, 241, 20)), fontsize=15)
    ax.set_zticklabels((np.arange(40, 221, 20)), fontsize=15)
    ax.legend(fontsize=20, loc='upper left')
    if 'cropped_faces' in df['Image Path'].iloc[0]:
        ax.set_title('Skin Tone Clusters for Fashion Dataset', fontsize=30)
        plt.savefig('./plots/skin_tone_clusters_fashion.png')
    else:
        ax.set_title('Skin Tone Clusters for LFW Dataset', fontsize=30)
        plt.savefig('./plots/skin_tone_clusters_lfw.png')
    plt.show()
    print('plotted clusters')

def plot_rgb_distributions(skintone_df: pd.DataFrame, image_dir: str) -> None:
    """
    Plot the R, G, and B distributions of the skin tones and save the plot.
    """
    plt.subplots(1, 3, figsize=(18, 6), sharey=True, dpi=300)
    for i, color in enumerate(['R', 'G', 'B']):
        print('color:\n', color)
        plt.subplot(1, 3, i + 1)
        sns.histplot(skintone_df[color], bins=30, color=color.lower())
        plt.title(f"{color} Channel")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
    if image_dir == 'cropped_faces':
        plt.suptitle('RGB Distributions for Fashion Dataset', fontsize=16)
        plt.savefig('./plots/rgb_distributions_fashion.png')
    else:
        plt.suptitle('RGB Distributions for LFW Dataset', fontsize=16)
        plt.savefig('./plots/rgb_distributions_lfw.png')
    plt.show()

def save_representatives(clustering_df: pd.DataFrame) -> None:
    """
    Save the representative images and their RGB values to a CSV file."""
    print('saving representatives')
    clustering_df = pd.DataFrame(clustering_df)
    clustering_df.loc[:, 'Representative_R'] = clustering_df.loc[:, 'Representative_R'] * 255
    clustering_df.loc[:, 'Representative_G'] = clustering_df.loc[:, 'Representative_G'] * 255
    clustering_df.loc[:, 'Representative_B'] = clustering_df.loc[:, 'Representative_B'] * 255
    clustering_df.loc[:, 'Cluster_R'] = clustering_df.loc[:, 'Cluster_R'] * 255
    clustering_df.loc[:, 'Cluster_G'] = clustering_df.loc[:, 'Cluster_G'] * 255
    clustering_df.loc[:, 'Cluster_B'] = clustering_df.loc[:, 'Cluster_B'] * 255
    rep_csv = clustering_df[['Image Path', 'Cluster', 'Representative', 'Representative_R', 'Representative_G', 'Representative_B', 'Cluster_R', 'Cluster_G', 'Cluster_B']]
    if 'cropped_faces' in clustering_df['Image Path'].iloc[0]:
        rep_csv.to_csv('./intermediate_files/representative_images_fashion.csv', index=False, header=True)
    else:
        rep_csv.to_csv('./intermediate_files/representative_images_lfw.csv', index=False, header=True)
    print('saved representatives')

def clustering_bias():
    '''Measures the bias in the clustering process by checking the distribution of skin tones in the clusters.'''

    # X = df[['R', 'G', 'B']].values
    # kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    # df['cluster'] = kmeans.labels_

    # df['cluster'].value_counts(normalize=True)

    #staticial parity 
    #parity = pd.Series(clusters).value_counts(normalize=True).sort_index()

    #training data bias - check distribution in raw image data
    return 


def error_rate():
    '''determine how often Morphe foundation shade reccomendations compare to model skin tone output
    1. Clustering gives us RGB values of representative image and 5 closest images (what K means thinks are the 5 closeest). 
    2. Morphe models (used in the AI tool) have an associated RGB  and also give 5 reccoemdnations 
    3. Pass the representative image into the 'I to'l'and e  what are the recceomendations (get 5 RGB)
    4. Metric 1 Jaccard similairity - how much do the representative image recceomdnations deviate from the morphe reccoemndatins (ie what is the percentage of overlapping shades - jaccard similarity)R    5. Metric 2 Statistical Parity - did the Morphe AI tool predict disappropriantely towards certain skintone groups (ie. visualize all the AI outputs, was it mostly fair toned, dark toned, etc - for this we have to check KMeans its self is biased)    '''
    pass

def main() -> None:
    """
    Main function to run the clustering and visualization process."""
    #image_dir = 'data/lfw-deepfunneled/'
    image_dir = 'cropped_faces'
    if image_dir == 'data/lfw-deepfunneled/':
        image_paths = get_image_paths(image_dir)
    else:
        image_paths = [os.path.join(image_dir, filename) for filename in \
                       os.listdir(image_dir) if filename.endswith(('.jpg', '.jpeg', '.png'))]
    skin_tones = process_images(image_paths)
    labels, avg_skin_tones, representatives, n_clusters = cluster_images(skin_tones, image_paths, n_clusters=10)
    plot_cluster_sizes(labels, avg_skin_tones, image_paths, n_clusters)
    clustering_df, skintone_df = process_img_for_clustering(pd.DataFrame(avg_skin_tones, columns=['R', 'G', 'B']), representatives)
    plot_rgb_distributions(skintone_df, image_dir)
    save_representatives(clustering_df)
    plot_clusters(clustering_df)

if __name__ == "__main__":
    main()