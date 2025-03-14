import cv2
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import random
import colorsys
import matplotlib.pyplot as plt
import seaborn as sns

def extract_skin(image):
    """Extract skin region using HSV color segmentation"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 40, 50], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin = cv2.bitwise_and(image, image, mask=mask)
    return skin

def get_dominant_color(image, k=1):
    """Find the dominant color in the image using K-means"""
    pixels = image.reshape(-1, 3)
    pixels = pixels[pixels.sum(axis=1) > 0]  # Remove black (masked) pixels
    if len(pixels) == 0:
        return np.array([0, 0, 0])  # Fallback for empty masks
    
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_[0]

def get_image_paths(root_dir):
    """Retrieve image paths from nested subdirectories"""
    image_paths = []
    for person_folder in os.listdir(root_dir):
        person_path = os.path.join(root_dir, person_folder)
        if os.path.isdir(person_path):  # Ensure it's a directory
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(img_path)
    image_paths = random.sample(image_paths, 10)  # Randomly sample 1000 images
    return image_paths

def process_images(image_paths):
    """Process images to extract dominant skin tone"""
    skin_tones = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        skin = extract_skin(img)
        dominant_color = get_dominant_color(skin)
        skin_tones.append(dominant_color)
    return np.array(skin_tones)

def cluster_images(skin_tones, image_paths, n_clusters=10):
    """Cluster images based on skin tones"""
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(skin_tones)

    # Get cluster centers (average skin tone)
    cluster_centers = kmeans.cluster_centers_

    # Find 3-4 representative images for each cluster
    representative_images = {}
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_skin_tones = skin_tones[cluster_indices]
        distances = np.linalg.norm(cluster_skin_tones - cluster_centers[i], axis=1)
        closest_indices = cluster_indices[np.argsort(distances)[:4]]
        representative_images[i] = [image_paths[idx] for idx in closest_indices]

    return labels, cluster_centers, representative_images

# Set the correct image directory
image_dir = 'data/lfw-deepfunneled/'

# Retrieve image paths
image_paths = get_image_paths(image_dir)

# Extract skin tones
skin_tones = process_images(image_paths)

# Cluster images and find representatives
labels, avg_skin_tones, representatives = cluster_images(skin_tones, image_paths, n_clusters=10)

print("Cluster Centers (Avg Skin Tones):", avg_skin_tones)
print("Representative Images:", representatives)

# plot the clusters
 
def plot_clusters(df):
    # we have a df with R, G, B and cluster
    # 10 rows
    # we want to plot the average skin tone for each cluster and color the points as the skin tones they represent
    # 3d scatterplot with R, G, B as the axes
    # normalize the values to 0-1
    df.loc[:, 'Cluster'] = df.index
    df.loc[:, 'R'] = df.loc[:, 'R'] / 255
    df.loc[:, 'G'] = df.loc[:, 'G'] / 255
    df.loc[:, 'B'] = df.loc[:, 'B'] / 255
    print(df.loc[:, ['R', 'G', 'B']].to_dict())
    df.loc[:, 'RGB_tuple'] = df.loc[:, ['R', 'G', 'B']].apply(tuple, axis=1)
    print(df.loc[:, 'RGB_tuple'].to_list())
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    color_map = df.loc[:, ['Cluster', 'RGB_tuple']].set_index('Cluster').to_dict()['RGB_tuple']
    print(color_map)
    colors = [color_map[c] for c in df['Cluster']]
    ax.scatter(df.loc[:, 'R'], df.loc[:, 'G'], df.loc[:, 'B'], c=colors, s=100)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title('Skin Tone Clusters')
    plt.savefig('skin_tone_clusters.png')
    plt.show()


plot_clusters(pd.DataFrame(avg_skin_tones, columns=['R', 'G', 'B']))
