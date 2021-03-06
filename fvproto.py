import tensorflow as tf
import numpy as np
from scipy import spatial

path1 = "images/tumblr1.jpg"
path2 = "images/tumblr2.jpg"

print(f"Load TensorFlow model: ")
model = tf.saved_model.load("hubmodule/feature-vector.4") #Loaded from https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, 224, 224)
    img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    return img

def fvhash(im):
    """
    https://towardsdatascience.com/image-similarity-detection-in-action-with-tensorflow-2-0-b8d9a78b2509
    :return:
    """
    features = model(im)
    return np.squeeze(features)

def __sub__(self, other):
    return spatial.distance.cosine(self, other)

im1 = load_img(path1)
im2 = load_img(path2)
h1 = fvhash(im1)
h2 = fvhash(im2)
print(h1)
print(h2)
print(__sub__(h1, h2))

