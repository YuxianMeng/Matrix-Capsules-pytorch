# Matrix-Capsules-pytorch
This is a pytorch implementation of [Matrix Capsules with EM routing](https://openreview.net/pdf?id=HJWLfGWRb)

In ```Capsules.py```, there are two implemented classes: ```PrimaryCaps``` and ```ConvCaps```.
The ClassCapsules in the paper is actually a special case of ```ConvCaps``` with whole receptive field, transformation matrix sharing and Coordinate Addition.

In ```train.py```, I define a CapsNet in the paper using classes in ```Capsules.py```, and could be used to train a model for MNIST dataset.

## Train a small CapsNet on MNIST
```python train.py -batch_size=64 -lr=2e-2 -num_epochs=5 -r=1 -print_freq=5```.

Note:
* more args can be found in ```utils.py```, and if you want to change A,B,C,D, go to ```line 62``` of ```train.py```
* m and lambda schedule need to be changed if you want to train a capsnet with r=2 or 3. The default schedule make capsnet does not converge in those cases.


## Results
The test accuracy is around 97.6% after 1 epoch and 98.7% after 2 epochs of training with a small Capsule of A,B,C,D,r = 64,8,16,16,1. After 30 epochs of training, the best acc is around 99.3%. More results on different configurations are welcomed.


## TODO
* using more matrix operation rather than ```for``` iteration in E-step of ```Capsules.py```.
* make capsules work when height_in != width_in
* find better lambda/m schedule to speed up the convergence.



