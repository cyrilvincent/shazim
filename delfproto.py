import numpy as np
import tensorflow as tf
from skimage.measure import ransac
from skimage.transform import AffineTransform
from scipy import spatial
from PIL import Image

delfmodel = tf.saved_model.load("hubmodule/delf.1").signatures['default']

path1 = "images/ski.jpg"
path2 = "images/ski2.jpg"

def delf(im):
    """
    https://www.tensorflow.org/hub/tutorials/tf_hub_delf_module
    """
    np_image = np.array(im)
    float_image = tf.image.convert_image_dtype(np_image, tf.float32)
    return delfmodel(
        image=float_image,
        score_threshold=tf.constant(100.0),
        image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
        max_feature_num=tf.constant(1000))

def delfsub(h1, h2):
    num_features_1 = h1['locations'].shape[0]
    num_features_2 = h2['locations'].shape[0]
    d1_tree = spatial.cKDTree(h1['descriptors'])
    _, indices = d1_tree.query(h2['descriptors'], distance_upper_bound=0.8)
    locations_2_to_use = np.array([
        h2['locations'][i,]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        h1['locations'][indices[i],]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    inliers = np.array([False])
    try:
        _, inliers = ransac(
            (locations_1_to_use, locations_2_to_use),
            AffineTransform,
            min_samples=3,
            residual_threshold=20,
            max_trials=1000)
    except:
        pass
    return inliers

im1 = Image.open(path1)
im2 = Image.open(path2)

h1 = delf(im1)
h2 = delf(im2)

res = delfsub(h1, h2)
print(len(res))
print(len(res[res]))






