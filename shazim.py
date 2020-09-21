import tensorflow as tf
import os
import imagehash
import numpy as np
import time
import argparse
import cyrilload
from scipy import spatial
from absl import logging
from typing import Dict
from PIL import Image

class ShazimEntity:

    def __init__(self, path):
        self.path = path
        self.ah = None
        self.dh = None
        self.fv = None

    def __sub__(self, other):
        res = {}
        res["dah"] = round(1 - (self.ah - other.ah) / len(self.ah.hash) ** 2, 3)
        res["ddh"] = round(1 - (self.dh - other.dh) / len(self.dh.hash) ** 2, 3)
        res["dfv"] = round(1 - spatial.distance.cosine(self.fv, other.fv), 3)
        return res

    def __repr__(self):
        return f"{self.path}"

class ShazimService:

    fvmodel = None

    def __init__(self, path):
        self.path = path
        if ShazimService.fvmodel == None:
            logging.set_verbosity(logging.ERROR)
            print(f"Load TensorFlow model: ")
            ShazimService.fvmodel = tf.saved_model.load("hubmodule/feature-vector.4")
        self.pil = Image.open(path)
        self.size = os.stat(path)[6]
        self.tfimg = self.load_img(self.path)

    def ah(self):
        """
        #http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
        :return:
        """
        return imagehash.average_hash(self.pil)

    def dh(self):
        """
        #http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
        :return:
        """
        return imagehash.dhash(self.pil)

    def fv(self):
        """
        https://towardsdatascience.com/image-similarity-detection-in-action-with-tensorflow-2-0-b8d9a78b2509
        :return:
        """
        features = ShazimService.fvmodel(self.tfimg)
        return np.squeeze(features)

    def load_img(self, path):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize_with_pad(img, 224, 224)
        img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        return img

class ShazimEngine:
    """
    Image parser and indexer
    Not thread safe
    """

    def __init__(self):
        self.db:Dict[str, ShazimEntity] = {}
        self.path = None

    def parse(self, path:str)->None:
        """
        Parser
        :param path: TXT file to parse
        """
        self.path = path
        self.nbi = 0
        t = time.perf_counter()
        self._parse(path)
        print(f"Found {self.nbi} images in {time.perf_counter() - t:.1f} s")

    def _parse(self, path):
        print(f"Parse {path}")
        for file in os.listdir(path):
            name = os.path.join(path, file)
            if os.path.isfile(name):
                file = file.upper()
                if ".JPG" in file or ".GIF" in file or ".PNG" in file or ".JPEG" in file or ".BMP" in file or ".SVG" in file:
                    try:
                        im = ShazimEntity(name)
                        self.db[name] = im
                        self.nbi += 1
                    except:
                        pass
            if os.path.isdir(name):
                self.parse(name)


    def save(self):
        cyrilload.save(self.db, "db")

    def load(self):
        self.db = cyrilload.load("db.pickle")
        self.nbi = len(self.db)

    def train(self):
        print(f"Hashing")
        t = time.perf_counter()
        i = 0
        for k in self.db.keys():
            if i % max(10,int(self.nbi / 100)) == 0:
                print(f"Hash {i + 1}/{self.nbi} in {time.perf_counter() - t:.1f} s")
            im = self.db[k]
            try:
                self.h_image(im)
            except Exception as ex:
                print(f"Error with {im}: {ex}")
            i+=1
        print(f"Hashed in {time.perf_counter() - t:.1f} s")

    def load_image(self, path):
        im = ShazimEntity(path)
        self.h_image(im)
        return im

    def h_image(self, im):
        service = ShazimService(im.path)
        im.ah = service.ah()  # 8x8
        im.dh = service.dh()  # 8x8
        im.fv = service.fv()  # 1280

    def predict(self, im1, im2, weigths=[1.0,1.0,2.0]):
        res = im1 - im2
        score = (res["ddh"] * weigths[0] + res["dah"] * weigths[1] + res["dfv"] * weigths[2]) / sum(weigths)
        return score

    def shazim(self, im, thresold=0.75, weigths=[1.0,1.0,2.0]):
        bests = []
        for k in self.db.keys():
            if im.path != k:
                im2 = self.db[k]
                score = self.predict(im, im2, weigths)
                if score > thresold:
                    bests.append((k, score))
        bests.sort(key = lambda x : x[1], reverse=True)
        return bests[:10]

if __name__ == '__main__':
    print("Shazim")
    print("======")
    parser = argparse.ArgumentParser(description="Shazim")
    parser.add_argument("path", help="Path")
    parser.add_argument("-p","--parse", action="store_true", help="Parse the path directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose and thresold to 0.5")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.parse: #-p images
        shazim = ShazimEngine()
        shazim.parse(args.path)
        shazim.train()
        shazim.save()
    else: #ski.jpg
        print("Load index")
        shazim = ShazimEngine()
        shazim.load()
        print(f"Load image {args.path}")
        im = shazim.load_image(args.path)
        print("Shazim...")
        thresold = 0.5 if args.verbose else 0.7
        res = shazim.shazim(im, thresold)
        print(f"Found {len(res)} image(s)")
        for i in res:
            print(f"{i[0]} at {i[1]*100:.0f}%")
            if args.verbose:
                diff = im - shazim.db[i[0]]
                print(diff)






