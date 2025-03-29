import numpy as np
import pandas as pd

# import morphe data
morphe_data = pd.read_csv('morphe_data.csv')
representative_data = pd.read_csv('representative_images.csv')
# for each representative image, get the 5 closest morphe label images
# we will compare the RGB values of each representative image and morphe label image, using min average distance as our metric
def get_closest_images(representative_data, morphe_data, n=5):
    out = pd.DataFrame(columns=['Representative_Image', 'Img1', 'Img2', 'Img3', 'Img4', 'Img5'])
    for index, row in representative_data.iterrows():
        representative_rgb = np.array([row['Representative_R'], row['Representative_G'], row['Representative_B']])
        morphe_rgb = np.array([morphe_data['r'], morphe_data['g'], morphe_data['b']]).T
        # min avg dist
        distances = []
        for i in range(len(morphe_rgb)):
            distance = np.mean(np.abs(representative_rgb - morphe_rgb[i]))
            distances.append(distance)
        closest_indices = np.argsort(distances)[:n]
        closest_images = morphe_data.iloc[closest_indices]
        closest_images = closest_images[['tone_name']].values.flatten()
        out.loc[index] = [row['Image Path']] + closest_images.tolist()
    print('out:\n', out)
    # Write to CSV after processing all images
    out.to_csv('closest_images.csv', index=False)
    return out

get_closest_images(representative_data, morphe_data, n=5)