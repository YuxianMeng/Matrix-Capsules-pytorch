# Matrix-Capsules-pytorch
This is a pytorch implementation of [Matrix Capsules with EM routing](https://openreview.net/pdf?id=HJWLfGWRb)

In ```Capsules.py```, there are two implemented classes: ```PrimaryCaps``` and ```ConvCaps```.
The ClassCapsules in the paper is actually a special case of ```ConvCaps``` with whole receptive field, transformation matrix sharing and Coordinate Addition.

In ```CapsNet.py```, I define a CapsNet in the paper using classes in ```Capsules.py```

## TODO
* run this model on a dataset
* using more matrix operation rather than ```for``` iteration.

