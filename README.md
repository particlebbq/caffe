# Caffe-subnet

This fork of caffe includes implementations of a few tools for reinforcement learning, generative adversarial 
networks, and variational autoencoders.  Some of the features available in this fork are:

- a Layer subclass (SubnetLayer) that wraps a Net object and exposes its inputs and outputs to the rest of the network
via the usual top and bottom blob vectors passed to Layer::Forward and Layer::Backward.  This can be used
for syntactical convenience in architectures with repeated network structures, e.g. DenseNet, 
[arXiv:1608.06993](https://arxiv.org/abs/1608.06993), to initialize different parts of a network with different
pre-trained models (no example use cases implemented right now, but they're not that hard to imagine) or as 
a component of other Layer sub-classes.
- A Layer subclass (UnrollLayer) that sets up a recurrent network given a .prototxt describing the network's structure
at one time slice.  (The stock Caffe implementation lets you do recurrent networks, but in my understanding you need to 
write a Layer subclass that constructs the unrolled network; the implementation in this fork makes it easier because you 
just have to work with .prototxt)
- A Layer subclass (AdversarialSubnetPairLayer) that sets up a generative adversarial network given .prototxt files describing
the generator and discriminator networks; see [arXiv:1406.2661](https://arxiv.org/abs/1406.2661).
This class handle the two-phase training of the adversarial networks by means of an internal counter, so it doesn't
need a special Solver class.
- A Layer subclass (VAELayer) that sets up a variational autoencoder given .prototxt files that describe the encoder and decoder networks;
see [arXiv:1312.6114](https://arxiv.org/abs/1312.6114).
- A few Layer subclasses that handle some stochastic operations for reinforcement learning, variational autoencoders, etc.  For example,
GaussianSampleLayer samples from a Gaussian centered around the input during the forward pass.

A few working examples are included in the examples/mnist directory:
- examples/mnist/solver_GAN_1FC.prototxt and examples/mnist/solver_GAN.prototxt implement generative adversarial networks, using fully-connected layers in the
first case and convolutional stacks in the second.  The script examples/mnist/gan_forward_mnist.py runs the trained network in 
the forward direction to produce a .png with a few generated samples.
- examples/mnist/solver_DRAM.prototxt trains a recurrent attention model like the one described in [arXiv:1412.7755](https://arxiv.org/abs/1412.7755)
to identify pairs of MNIST digits embedded into a larger (otherwise blank) image.  The script examples/mnist/evaluate_sumsamples_mnist_DRAM.py does the averaging 
described in equation 14 of the paper.
- examples/mnist/solver_VAE.prototxt trains a variational autoencoder to produce MNIST samples, following [arXiv:1312.6114](https://arxiv.org/abs/1312.6114).


(NB: this is not the official code for any of the above papers; it's an independent reimplementation.)


# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
