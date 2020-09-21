Shazim
===========

Shazam images

Based on ImageHash and Tensorflow with the MobileNet model

Rationale
---------

Shazim retrieve similar images with AI Deep Learning and advanced hash image algorithms

The AI hash use the MobileNet Tensorflow model
Image hash algorithms analyse the image structure with linear algebra

Shazim compares images hash to compute the distance between to images

Requirements
-------------
Based on Tensorflow, Pillow, numpy, scipy, ImageHash,
Easy installation

	pip install -r requirements.txt

Basic usage
------------

    shazim.py ski.jpg

    Found 4 image(s)
    images\ski.jpg at 100%
    images\ski_copy.jpg at 100%
    images\ski2.jpg at 84%
    images\000000037689.jpg at 75%


Original: <a href="ski.jpg"><img src="ski.jpg" height="100"/></a>
Copy 100%: <a href="images/ski_copy.jpg"><img src="images/ski_copy.jpg" height="100"/></a>
Flip 84%: <a href="images/ski2.jpg"><img src="images/ski2.jpg" height="100"/></a>
Other 75%: <a href="images/000000037689.jpg"><img src="images/000000037689.jpg" height="100"/></a>

Detect modified image from originals

    shazim.py lenna.jpg

    Found 3 image(s)
    images\lenna.png at 100%
    images\lenna-crop.jpg at 93%
    images\lenna1.jpg at 72%


Original: <a href="lenna.png"><img src="lenna.png" height="100"/></a>
Same 100%: <a href="images/lenna.png"><img src="images/lenna.png" height="100"/></a>
Cropped image 93%: <a href="images/lenna-crop.jpg"><img src="images/lenna-crop.jpg" height="100"/></a>
Text inserted 72%: <a href="images/lenna1.jpg"><img src="images/lenna1.jpg" height="100"/></a>

Avanced usage
--------------
Image directory must be parsed for training before Shazim
::

    shazim.py -p images

    Parse images
    Found 15 images in 0.0 s
    Hashing
    Hash 1/15 in 0.0 s
    Load TensorFlow model:
    Hash 11/15 in 3.2 s
    Hashed in 3.3 s
    Save db.pickle

Verbose detection
------------------
Thresold = 50% instead of 75%:
    - dah: AverageHash score (generate a 8x8 matrix with average points and compute the hamming distance)
    - ddh: Difference Hash (generate 8x8 matrix with Discret Cosine Transform from gradients and compute the hamming distance)
    - dfv: Feature Vector Hash (generate a 1280 vector from the output of the convulational parts of the MobileNet network and compute the cosine distance)
    - dsize: Image size difference
    - note that wavelets hash are not used beacause it's too slow and perceptual hash are not used because difference hash is better

::

    shazim.py -v ski.jpg

    Parse images
    Found 10 image(s)
    images\ski.jpg at 100%
    {'dah': 1.0, 'ddh': 1.0, 'dfv': 1.0, 'dsize': 0}
    images\ski_copy.jpg at 100%
    {'dah': 1.0, 'ddh': 1.0, 'dfv': 1.0, 'dsize': 0}
    images\ski2.jpg at 84%
    {'dah': 0.953, 'ddh': 0.719, 'dfv': 0.836, 'dsize': 363613}
    images\000000037689.jpg at 75%
    {'dah': 0.875, 'ddh': 0.578, 'dfv': 0.782, 'dsize': 320281}
    images\000000038118.jpg at 73%
    {'dah': 0.844, 'ddh': 0.609, 'dfv': 0.73, 'dsize': 373509}
    images\ski3.jpg at 71%
    {'dah': 0.828, 'ddh': 0.5, 'dfv': 0.754, 'dsize': 316568}
    images\cat.10994.jpg at 57%
    {'dah': 0.531, 'ddh': 0.422, 'dfv': 0.661, 'dsize': 465555}
    images\cat.11016.jpg at 57%
    {'dah': 0.625, 'ddh': 0.562, 'dfv': 0.544, 'dsize': 461972}
    images\00000005.jpg at 55%
    {'dah': 0.656, 'ddh': 0.375, 'dfv': 0.591, 'dsize': 468151}
    images\lenna1.jpg at 54%
    {'dah': 0.5, 'ddh': 0.5, 'dfv': 0.582, 'dsize': 443493}

API
---
Training:

    from shazim import ShazimEngine

    shazim = ShazimEngine()
    shazim.parse(args.path)
    shazim.train()
    shazim.save()

Predict:

    from shazim import ShazimEngine

    shazim = ShazimEngine()
    shazim.load()
    im = shazim.load_image("ski.jpg")
    thresold = 0.75
    res = shazim.shazim(im, thresold)

Compare two images:

    from shazim import ShazimEngine

    shazim = ShazimEngine()
    im1 = shazim.load_image("ski.jpg")
    im2 = shazim.load_image("ski_copy.jpg")
    res = im1 - im2

Hash image

    from shazim import ShazimEngine

    shazim = ShazimEngine()
    im = shazim.load_image("ski.jpg")
    im.ah #Average hash
    im.dh #Difference hash
    im.fv #MobileNet hash

Sources hosted at GitHub: https://github.com/CyrilVincent/shazim

http://www.CyrilVincent.com

Links
------
https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4

https://pypi.org/project/ImageHash/



