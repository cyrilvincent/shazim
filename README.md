Shazim
===========

Shazam images

Based on Tensorflow with the MobileNet model and ImageHash 

Rationale
---------

Shazim retrieve similar images with AI Deep Learning and advanced hash image algorithms

The AI analyse images with MobileNet Tensorflow neural network then image structure are analysed with linear algebra

Shazim compares images hash to compute the distance between 2 images

Requirements
-------------
Based on Tensorflow, Pillow, Numpy, Scipy, ImageHash,

Easy installation, tested with Python 3.7 & Tensorflow 2.3

	pip install -r requirements.txt
	python shazim.py ski.jpg

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

    # dah: AverageHash score (generate a 8x8 matrix with average points and compute the hamming distance)
    # ddh: Difference Hash (generate 8x8 matrix with Discret Cosine Transform from gradients and compute the hamming distance)
    # dfv: Feature Vector Hash (generate a 1280 vector from the output of the convulational parts of the MobileNet network and compute the cosine distance)
    # dsize: Image size difference
    # Wavelets hash are not used beacause it's too slow and perceptual hash are not used because difference hash is better
    # VGG*, Inception*, ResNet* are not used because the training is too slow
    
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
    shazim.parse("images")
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
    
Understand Image Hashing vs Deep Learning

Image hashing compute a 8x8 matrix and detect similar image or photoshoped image

For Image Hashing those images are the same :
<a href="images/forest-high.jpg"><img src="images/forest-high.jpg" height="100"/></a> <a href="images/forest-copyright.jpg"><img src="images/forest-copyright.jpg" height="100"/></a>
    
    shazim.py images\forest-high.jpg
    
    Shazim...
    Found 1 image(s)
    images\forest-copyright.jpg at 98%

And those images are differents :
<a href="images/tumblr1.jpg"><img src="images/tumblr1.jpg" height="100"/></a> <a href="images/tumblr2.jpg"><img src="images/tumblr2.jpg" height="100"/></a>

    shazim.py images\tumblr1.jpg
    
    Shazim...
    Found 0 image(s)

Deep Learning wants to detect that images are the same

Let see the prediction with -v option :

    shazim.py images\tumblr1.jpg -v
    
    Found 10 image(s)
    images\tumblr2.jpg at 70%
    {'dah': 0.609, 'ddh': 0.453, 'dfv': 0.86, 'dsize': 2596}
    # dh : Difference hash doest not detect anything
    # ah : Average hash detect only 61% similarity
    # fv : MobileNet detect well at 86%
    
    #Ponderation between hash methodes are w = [1.0,1.0,2.0]

How to changes ponderation betweens models :

    from shazim import ShazimEngine

    shazim = ShazimEngine()
    shazim.load()
    im = shazim.load_image("ski.jpg")
    thresold = 0.75
    weights = [1.0,1.0,2.0] #defaults
    res = shazim.shazim(im, thresold, weights)

How to code this with Deep Learning only

    from shazim import ShazimEngine

    shazim = ShazimEngine()
    im1 = shazim.load_image("tumblr1.jpg")
    im2 = shazim.load_image("tumblr2.jpg")
    res = im1 - im2
    res["dfv"]
    
    # or
    
    weights = [0.0,0.0,1.0] #ad, dh, fv
    shazim.predict(im1, im2, weights)
    
How to code this with Image Hash only

    from shazim import ShazimEngine

    shazim = ShazimEngine()
    im1 = shazim.load_image("tumblr1.jpg")
    im2 = shazim.load_image("tumblr2.jpg")
    res = im1 - im2
    res["dah"] #or res["dah"]
    
    # or
    
    weights = [0.5,0.5,0.0] #ad, dh, fv
    shazim.predict(im1, im2, weights)
    



Sources hosted at GitHub: https://github.com/CyrilVincent/shazim

http://www.CyrilVincent.com

Links
------
https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4

https://pypi.org/project/ImageHash/



