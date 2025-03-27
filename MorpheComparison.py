import cv2
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# import morphe data
morphe_data = pd.read_csv('morphe_data.csv')

# for each representative image, get the 5 closest morphe label images

# return those 5 image labels