import cv2
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# import morphe data
morphe_data = pd.read_csv('morphe_data.csv')
representative_data = pd.read_csv('representative_images.csv')
representative_data.loc[:, 'Representative_R'] = representative_data.loc[:, 'Representative_R'] * 255
representative_data.loc[:, 'Representative_G'] = representative_data.loc[:, 'Representative_G'] * 255
representative_data.loc[:, 'Representative_B'] = representative_data.loc[:, 'Representative_B'] * 255

# for each representative image, get the 5 closest morphe label images
# we will compare the RGB values of each representative image and morphe label image, using min average distance as our metric
def get_closest_images(representative_data, morphe_data, n=5):
    print('representative_data:\n', representative_data.head(10))
    print('morphe_data:\n', morphe_data.head())
    out = pd.DataFrame(columns=['Representative_Image', 'Img1', 'Img2', 'Img3', 'Img4', 'Img5'])
    for index, row in representative_data.iterrows():
        print('row:\n', row)
        representative_rgb = np.array([row['Representative_R'], row['Representative_G'], row['Representative_B']])
        morphe_rgb = np.array([morphe_data['r'], morphe_data['g'], morphe_data['b']]).T
        distances = np.linalg.norm(morphe_rgb - representative_rgb, axis=1)
        closest_indices = np.argsort(distances)[:n]
        closest_images = morphe_data.iloc[closest_indices]
        print('closest_indices:\n', closest_indices)
        print('closest_images:\n', closest_images)
        closest_images = closest_images[['tone_name']].values.flatten()
        out.loc[index] = [row['Image Path']] + closest_images.tolist()
        print('closest_images:\n', closest_images)
        print('out:\n', out)
        out.to_csv('closest_images.csv', index=False)

get_closest_images(representative_data, morphe_data, n=5)