# Awesome hypernetworks

A curated list of awesome hypernetwork resources, inspired by [awesome implicit representations]
(https://github.com/vsitzmann/awesome-implicit-representations).

## Introduction
Hpernetworks have become common in the field of deep learning and appear in some
way or another already in thousands of papers. In the following I will therefore try to make a list of resources that are *only* a good representative of the most interesting concepts around
HyperNetworks. Also, there will be bias towards papers of mine. 

Please get in touch when you think I missed important references.

## What are Hypernetworks?

HyperNetworks are simply neural networks that produce and/or adapt parameters of another parametrized model.
Without surprise, they at least date back to the beginning of 1990s and to Schmidhuber in the
context of [meta-learning](https://people.idsia.ch/~juergen/metalearning.html#FASTMETA1) and
[self-referential](https://people.idsia.ch/~juergen/metalearner.html).
Hypernetworks have been applied in a very large range of deep learning contexts and applications
which I try to cover below.

# Papers
# Adaptive Layers
The core idea of *adaptive layers* is the make the parameters of a certain layer of the neural
network adapt to computation that preceded the layers computation. Usually, and in contrast to that, the computational node of parameters have no parent and the node’s value is static during the
“forward” computation. As an example, we can think of
s_{t+1} = phi(s_t)*W + U*x_t
as a quite general form of a neural network computation of layer t+1. A simple *adaptive layer*
would be, for example, 
s_{t+1} = phi(s_t)*W_t + U*x_t
with W_t = W_t + H*s_t with H a 3-dim Tensor of size |s_t| that we can train with gradient descent
instead.


https://people.idsia.ch/~juergen/FKI-147-91ocr.pdf
https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html
https://www.semanticscholar.org/paper/Evolving-Modular-Fast-Weight-Networks-for-Control-
Gomez-Schmidhuber/9382c0ec9904ea93089f439479607f7fd0195505
https://icml.cc/Conferences/2011/papers/524_icmlpaper.pdf
https://arxiv.org/pdf/1609.09106.pdf
https://arxiv.org/abs/1606.02580https://www.robots.ox.ac.uk/~vgg/publications/2016/BertinettoFC16/bertinettofc16.pdf
https://arxiv.org/abs/1605.09673
https://arxiv.org/abs/1902.09701
https://arxiv.org/abs/1709.07871
https://openaccess.thecvf.com/content_cvpr_2015/papers/
Klein_A_Dynamic_Convolutional_2015_CVPR_paper.pdf
https://proceedings.neurips.cc/paper/2016/file/9f44e956e3a2b7b5598c625fcc802c36-Paper.pdf
https://openreview.net/forum?id=mLcmdlEUxy-
https://papers.nips.cc/paper/2013/file/7fec306d1e665bc9c748b5d2b99a6e97-Paper.pdf
https://openreview.net/forum?id=rylnK6VtDH
https://proceedings.neurips.cc/paper/2017/file/e7e23670481ac78b3c4122a99ba60573-Paper.pdf
https://arxiv.org/abs/1910.01526
## Self-attention
Self-attention are a form of adaptive layers. Nevertheless, I will not cover transformer literature
here but only a couple of papers that use self-attention in a different way.
https://arxiv.org/abs/2102.11174
https://arxiv.org/abs/1901.10430
https://arxiv.org/abs/2003.08165
# Architecture search and Hypernetworks used in Neuroevolution
https://arxiv.org/abs/1708.05344
https://arxiv.org/pdf/1810.05749.pdf
https://ieeexplore.ieee.org/document/6792316
https://arxiv.org/pdf/1606.02580.pdf
https://people.idsia.ch/~juergen/gecco2010koutnik.pdf
https://link.springer.com/chapter/10.1007/978-3-030-30493-5_52
# Implicit Neural Representations
Implicit Neural Representations are continuous functions, usually neural networks, that simply
represent a map between a domain and the signal value. In order
https://arxiv.org/abs/1908.06277
https://openaccess.thecvf.com/content/CVPR2021/papers/
Skorokhodov_Adversarial_Generation_of_Continuous_Images_CVPR_2021_paper.pdf
https://arxiv.org/abs/1812.03828
https://arxiv.org/pdf/2006.09661.pdf
https://people.idsia.ch/~juergen/gecco2010koutnik.pdf
https://arxiv.org/pdf/1906.01618.pdf
# Meta- and Continual Learning
Algorithms that tackle meta- and continual learning with the help of hypernetworks been
extensively developed. Naturally, one can view the considered problems as acting on different time
scales and formulate them as solutions to a bilevel optimization or related formulations where
Hypernetworks can shine.
https://meta-learn.github.io/2019/papers/metalearn2019-flennerhag.pdf
https://arxiv.org/abs/1606.04474
https://arxiv.org/abs/2007.02933https://arxiv.org/abs/1801.05558
https://meta-learn.github.io/2020/papers/38_paper.pdf
https://arxiv.org/abs/1810.03642
https://arxiv.org/abs/1909.00025
https://arxiv.org/pdf/1906.07697.pdf
https://arxiv.org/abs/1807.05960
https://arxiv.org/abs/1906.00695
https://arxiv.org/abs/2006.12109
https://papers.nips.cc/paper/2021/file/ac796a52db3f16bbdb6557d3d89d1c5a-Paper.pdf
https://arxiv.org/abs/1802.09419
https://arxiv.org/pdf/1903.03088.pdf
# Reinforcement learning
I have not seen many papers so far that use hypernetworks to tackle RL problems explicitly.
Nevertheless,
https://arxiv.org/abs/2006.07464
https://ieeexplore.ieee.org/document/9560793
https://arxiv.org/pdf/2106.06842.pdf
# Modeling distributions
The following papers use hypernetworks to model a distribution over the weights of the target
network. For example, one can use a hypernetwork to transform a simple normal distribution into a
potentially complex weight distribution that captures the epistemic uncertainty of the model.
https://arxiv.org/abs/2111.11763
https://arxiv.org/pdf/1810.00555.pdf
https://arxiv.org/abs/1711.01297
https://arxiv.org/abs/1710.04759
https://arxiv.org/pdf/1703.01961.pdf
https://arxiv.org/abs/2007.12927
https://www.zora.uzh.ch/id/eprint/168578/
https://arxiv.org/abs/2002.04033
http://proceedings.mlr.press/v80/huang18d/huang18d.pdf
# Others
https://openreview.net/pdf?id=ryup8-WCW
https://arxiv.org/pdf/2002.10006.pdf
https://arxiv.org/pdf/2003.12193.pdf
https://openreview.net/pdf?id=H1lma24tPB
https://arxiv.org/pdf/1704.03477.pdf
https://openaccess.thecvf.com/content_cvpr_2018/html/
Shen_Neural_Style_Transfer_CVPR_2018_paper.html

