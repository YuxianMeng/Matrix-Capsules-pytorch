# Matrix-Capsules-pytorch
This is a pytorch implementation of [Matrix Capsules with EM routing](https://openreview.net/pdf?id=HJWLfGWRb)

In ```Capsules.py```, there are two implemented classes: ```PrimaryCaps``` and ```ConvCaps```.
The ClassCapsules in the paper is actually a special case of ```ConvCaps``` with whole receptive field, transformation matrix sharing and Coordinate Addition.

In ```train.py```, I define a CapsNet in the paper using classes in ```Capsules.py```, and could be used to train a model for MNIST dataset.

## Train a small CapsNet on MNIST
```python train.py -batch_size=64 -use_cuda=True -lr=2e-2 -num_epochs=5 -r=3```



## TODO
* using more matrix operation rather than ```for``` iteration in E-step of ```Capsules.py```.

