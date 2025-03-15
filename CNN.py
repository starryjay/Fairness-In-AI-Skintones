import cv2
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import random
import colorsys
import matplotlib.pyplot as plt
import seaborn as sns

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
    return image_paths

def process_images(image_paths):
    skin_tones = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        skin = extract_skin(img)
        dominant_color = get_dominant_color(skin)
        skin_tones.append(dominant_color)
    return np.array(skin_tones)

def cluster_images(skin_tones, image_paths, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(skin_tones)
    cluster_centers = kmeans.cluster_centers_
    representative_images = {}
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_skin_tones = skin_tones[cluster_indices]
        distances = np.linalg.norm(cluster_skin_tones - cluster_centers[i], axis=1)
        closest_indices = cluster_indices[np.argsort(distances)[:4]]
        representative_images[i] = [image_paths[idx] for idx in closest_indices]
    return labels, cluster_centers, representative_images

image_dir = 'data/lfw-deepfunneled/'
image_paths = get_image_paths(image_dir)
skin_tones = process_images(image_paths)
labels, avg_skin_tones, representatives = cluster_images(skin_tones, image_paths, n_clusters=10)

print("Cluster Centers (Avg Skin Tones):", avg_skin_tones)
print("Representative Images:", representatives)
 
def plot_clusters(df):
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