import cv2
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# get cluster centers for grouped skin tone values
# get rgb values of each morphe label image
# get the 4 representative images from each cluster that are closest to the centers
# get difference between representative images and the morphe label images
    # get 5 closest morphe label images to each representative image
# test representative images on the morphe algorithm and see which of their images they match to
# get difference in skintones between the labels we calculated and the labels from the morphe algo

def extract_skin(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 40, 50], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin = cv2.bitwise_and(image, image, mask=mask)
    return skin

def get_dominant_color(image, k=1):
    pixels = image.reshape(-1, 3)
    pixels = pixels[pixels.sum(axis=1) > 0]
    if len(pixels) == 0:
        return np.array([0, 0, 0])
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_[0]

def get_image_paths(root_dir):
    image_paths = []
    for person_folder in os.listdir(root_dir):
        person_path = os.path.join(root_dir, person_folder)
        if os.path.isdir(person_path):
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(img_path)
    print('got image paths')
    return image_paths

def process_images(image_paths): 
    if 'skin_tones.csv' in os.listdir('.'):
        skin_tones_df = pd.read_csv('skin_tones.csv')
        skin_tones = skin_tones_df.set_index('Image Path').T.to_dict()
    else:
        skin_tones = {}
        for path in image_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            skin = extract_skin(img)
            dominant_color = get_dominant_color(skin)
            if dominant_color is None or np.sum(dominant_color) == 0:
                continue
            else:
                skin_tones[path] = dominant_color
        skin_tones_df = pd.DataFrame.from_dict(skin_tones, orient='index', columns=['R', 'G', 'B'])
        skin_tones_df.to_csv('skin_tones.csv', index=True, header=True, index_label='Image Path')
    print('processed images')
    return skin_tones

def cluster_images(skin_tones, image_paths, n_clusters=10):
    # extract each dict from skin_tones, convert the three values to a tuple, add tuple to list
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
    return labels, cluster_centers, representative_images

image_dir = 'data/lfw-deepfunneled/'
image_paths = get_image_paths(image_dir)
skin_tones = process_images(image_paths)
labels, avg_skin_tones, representatives = cluster_images(skin_tones, image_paths, n_clusters=10)

def process_img_for_clustering(df, representatives):
    print('plotting clusters')
    skin_tones = {}
    for cluster, img_list in representatives.items():
        skin_tones[cluster] = [(img_path, process_images(img_list)[img_path]) for img_path in img_list]
    skintone_df = pd.DataFrame.from_dict(skin_tones, orient='index')
    # split each column into img path, r, g, b
    skintone_df = skintone_df.stack().reset_index()
    skintone_df.columns = ['Cluster', 'Representative', 'RGB']
    # scale cluster from 1-10
    skintone_df['Cluster'] = skintone_df['Cluster'].astype(int) + 1
    print(skintone_df['Cluster'].unique())
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
    df = pd.merge(df, skintone_df, on='Cluster', how='left').rename(columns={'R_x': 'Cluster_R', 'G_x': 'Cluster_G', 'B_x': 'Cluster_B', 
                                                                             'R_y': 'Representative_R', 'G_y': 'Representative_G', 'B_y': 'Representative_B', 
                                                                             'RGB_tuple_x': 'Cluster_RGB_tuple', 'RGB_tuple_y': 'Representative_RGB_tuple'})
    return df

def plot_clusters(df):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    markers = ['o', 's', 'd', '^', '8', 'P', 'X', 'p', '*', 'h']
    unique_clusters = df['Cluster'].unique()
    cluster_marker_map = {cluster: markers[i % len(markers)] for i, cluster in enumerate(unique_clusters)}
    cluster_color_map = df.loc[:, ['Cluster', 'Cluster_RGB_tuple']].set_index('Cluster').to_dict()['Cluster_RGB_tuple']
    print(cluster_color_map)
    for cluster in unique_clusters:
        cluster_data = df[df['Cluster'] == cluster]
        ax.scatter(cluster_data['Cluster_R'], cluster_data['Cluster_G'], cluster_data['Cluster_B'], 
                   c=[cluster_color_map[cluster]], 
                   s=320, 
                   marker=cluster_marker_map[cluster], 
                   edgecolors='black',
                   linewidth=0.8,
                   label=f'Cluster {cluster}')
    for cluster in unique_clusters:
        rep_data = df[df['Cluster'] == cluster]
        ax.scatter(rep_data['Representative_R'], rep_data['Representative_G'], rep_data['Representative_B'], 
                   c=[cluster_color_map[cluster]], 
                   s=200,
                   marker=cluster_marker_map[cluster]
                   )
    for i, row in df.iterrows():
        ax.text(row['Cluster_R'], row['Cluster_G'], row['Cluster_B'], str(row['Cluster']), size=15, zorder=100)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title('Skin Tone Clusters')
    ax.legend(fontsize=15)
    plt.savefig('skin_tone_clusters.png')
    plt.show()

clustering_df = process_img_for_clustering(pd.DataFrame(avg_skin_tones, columns=['R', 'G', 'B']), representatives)
plot_clusters(clustering_df)