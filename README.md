# Awesome hypernetworks

A curated list of awesome hypernetwork resources, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision) and [awesome implicit representations](https://github.com/vsitzmann/awesome-implicit-representations).

## Introduction
Hypernetworks have become very common in the field of deep learning and appear in some
way or another already in thousands of papers. In the following, I will therefore try to make a list of resources that are *only* a good representative of the most interesting concepts around
HyperNetworks. Also, there will be bias towards my papers. 

Please get in touch when you think I missed important references.

## What are Hypernetworks?

HyperNetworks are simply neural networks that produce and/or adapt parameters of another parametrized model.
Without surprise, they at least date back to the beginning of the 1990s and Schmidhuber in the
context of [meta-learning](https://people.idsia.ch/~juergen/metalearning.html#FASTMETA1) and
[self-referential](https://people.idsia.ch/~juergen/metalearner.html).
Hypernetworks have been applied in a very large range of deep learning contexts and applications
which I try to cover below.

# Papers
## Adaptive Layers
The core idea of *adaptive layers* is the make the parameters of a certain layer of the neural
network adapt to computation that preceded the layers computation. Usually, and in contrast to that, the computational node of normal parameters have no parent and the node’s value is static during the
“forward” computation. 


Fast weights and work on RNNs.

- [Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks](https://people.idsia.ch/~juergen/FKI-147-91ocr.pdf)  (Schmidhuber 1991)
- [Evolving Modular Fast-Weight Networks for Control](https://www.researchgate.net/publication/225137557_Evolving_Modular_Fast-Weight_Networks_for_Control) (Gomez & Schmidhuber 2005)
- [Generating Text with Recurrent Neural Networks](https://icml.cc/Conferences/2011/papers/524_icmlpaper.pdf) (Sutskever et. al 2011)
- [HyperNetworks](https://arxiv.org/pdf/1609.09106.pdf) (Ha et. al 2016)
- [Using Fast Weights to Attend to the Recent Past](https://proceedings.neurips.cc/paper/2016/file/9f44e956e3a2b7b5598c625fcc802c36-Paper.pdf) (Ba et. al 2016)
- [Recurrent Independent Mechanims](https://openreview.net/pdf?id=mLcmdlEUxy-) (Goyal et. al 2021)

Work on CNNs. 

- [Predicting Parameters in Deep Learning](https://papers.nips.cc/paper/2013/file/7fec306d1e665bc9c748b5d2b99a6e97-Paper.pdf) (Denil et. al 2013)
- [A Dynamic Convolutional Layer for Short Range Weather Prediction](https://openaccess.thecvf.com/content_cvpr_2015/papers/Klein_A_Dynamic_Convolutional_2015_CVPR_paper.pdf) (Klein et. al 2015)
- [Dynamic Filter Networks](https://arxiv.org/abs/1605.09673) (De Brabandere et. al 2016)
- [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549) (Bertinetto et. al 2016)
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871) (Perez et. al 2017)
- [Incorporating Side Information by Adaptive Convolution](https://proceedings.neurips.cc/paper/2017/file/e7e23670481ac78b3c4122a99ba60573-Paper.pdf) (Kang et. al 2017)
- [Learning Implicitly Recurrent CNNs Through Parameter Sharing](https://arxiv.org/abs/1902.09701) (Savarese & Maire 2019)

Work on generative models. 
The following two papers simply condition the generators of a GAN on side information. Probably there is more interesting work, please contact me if you know of something. 
I also list my paper "continual learning with hypernetworks" here because we use a hypernetwork i.a. to generate weights of a decoder in a variational autoencoder. 

- [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096) (Brock et. al 2018)
- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) (Karras et. al 2018)
- [Continual learning with hypernetworks](https://arxiv.org/abs/1906.00695) (von Oswald et. al 2020)

An overview of multiplicative interactions and hypernetworks
 
- [Multiplicative Interactions and Where to Find Them ](https://openreview.net/forum?id=rylnK6VtDH) (Jayakumar et. al 2020)


## Self-attention
Self-attention is a form of adaptive layers. Nevertheless, I will not cover transformer literature
here but mention this Schlag, Irie and Schmidhuber paper that discusses the equivalence to fast weights:

- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) (Schlag et. al 2021)

## Architecture search and Hypernetworks used in Neuroevolution
There has been very nice ideas that use Hypernetworks in architecture search. This list is probably far from accurate and complete. 
- [A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks](https://ieeexplore.ieee.org/document/6792316) (Stanley et. al 2009)
- [Evolving Neural Networks in Compressed Weight Space](https://people.idsia.ch/~juergen/gecco2010koutnik.pdf) (Koutník et. al 2010)
- [Convolution by Evolution](https://arxiv.org/abs/1606.02580) (Fernando et. al 2016)
- [SMASH: One-Shot Model Architecture Search through HyperNetworks](https://arxiv.org/abs/1708.05344) (Brock et. al 2017 )
- [Graph HyperNetworks for Neural Architecture Search](https://arxiv.org/abs/1810.05749) (Zhang et. al 2019)

## Implicit Neural Representations
Implicit Neural Representations are continuous functions, usually neural networks, that simply
represent a map between a domain and the signal value. Interestingly, hypernetworks are used in this framework intensively. 

- [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828) (Mescheder et. al 2018)
- [Deep Meta Functionals for Shape Representation](https://arxiv.org/abs/1908.06277) (Littwin & Wolf 2019)
- [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661) (Sitzmann et. al 2020)
- [Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations](https://arxiv.org/abs/1906.01618) (Sitzmann et. al 2019)
- [Adversarial Generation of Continuous Images](https://openaccess.thecvf.com/content/CVPR2021/papers/Skorokhodov_Adversarial_Generation_of_Continuous_Images_CVPR_2021_paper.pdf) (Skorokhodov et. al 2021)
- [MetaAvatar: Learning Animatable Clothed Human Models from Few Depth Images](https://neuralbodies.github.io/metavatar/) (Wang et. al 2021)


## Meta- and Continual Learning
Algorithms that tackle meta- and continual learning with the help of hypernetworks have been developed extensively. Naturally, one can view the considered problems as acting on different time
scales and formulate them as solutions to a bilevel optimization or related formulations where
Hypernetworks can work well.

- [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474) (Andrychowicz et. al 2016)
- [Fast Context Adaptation via Meta-Learning](https://arxiv.org/abs/1810.03642) (Zintgraf et. al 2018)
- [Meta-Learning with Latent Embedding Optimization](https://arxiv.org/abs/1807.05960) (Requeima et. al 2018)
- [Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace](https://arxiv.org/abs/1801.05558) (Lee & Choi 2018)
- [Stochastic Hyperparameter Optimization through Hypernetworks](https://arxiv.org/abs/1802.09419) (Lorraine & Duvenaud et. al 2018)
- [Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes](https://arxiv.org/abs/1906.07697) (Requeima et. al 2019)
- [Meta-Learning with Warped Gradient Descent](https://arxiv.org/abs/1909.00025) (Flennerhag et. al 2019)
- [Self-Tuning Networks: Bilevel Optimization of Hyperparameters using Structured Best-Response Functions](https://arxiv.org/abs/1903.03088) (MacKay et. al 2019)
- [Meta-Learning Symmetries by Reparameterization](https://arxiv.org/abs/2007.02933) (Zhou et. al 2020)
- [Meta-Learning via Hypernetworks](https://meta-learn.github.io/2020/papers/38_paper.pdf) (Zhao et. al 2020)
- [Continual learning with hypernetworks](https://arxiv.org/abs/1906.00695) (von Oswald et. al 2020)
- [Continual Learning in Recurrent Neural Networks](https://arxiv.org/abs/2006.12109) (Ehret et. al 2020)
- [Meta Internal Learning](https://papers.nips.cc/paper/2021/file/ac796a52db3f16bbdb6557d3d89d1c5a-Paper.pdf) (Bensadoun et. al 2021)

## Reinforcement learning
I have not seen many papers so far that use hypernetworks to tackle RL problems explicitly. Please contact me if you know of any.

- [Hypermodels for Exploration](https://arxiv.org/abs/2006.07464) (Dwaracherla et. al 2020)
- [Continual Model-Based Reinforcement Learning with Hypernetworks](https://ieeexplore.ieee.org/document/9560793) (Huang et. al 2021)
- [Recomposing the Reinforcement Learning Building Blocks with Hypernetworks](https://arxiv.org/pdf/2106.06842.pdf) (Keynan et. al 2021)


## Modeling distributions
The following papers use hypernetworks to model a distribution over the weights of the target
network. For example, one can use a hypernetwork to transform a simple normal distribution into a
potentially complex weight distribution that captures the epistemic uncertainty of the model.

- [Implicit Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1711.01297) (Pawlowski et. al 2017)
- [Bayesian Hypernetworks](https://arxiv.org/abs/1710.04759) (Krueger et. al 2017)
- [Probabilistic Meta-Representations Of Neural Networks](https://arxiv.org/abs/1810.00555) (Karaletsos et. al 2018)
- [Neural Autoregressive Flows](http://proceedings.mlr.press/v80/huang18d/huang18d.pdf) (Huang et. al 2018)
- [Approximating the Predictive Distribution via Adversarially-Trained Hypernetworks](https://www.zora.uzh.ch/id/eprint/168578/) (Henning et. al 2018)
- [Neural networks with late-phase weights](https://arxiv.org/abs/2007.12927) (von Oswald et. al 2020)
- [Hierarchical Gaussian Process Priors for Bayesian Neural Network Weights](https://arxiv.org/abs/2002.04033) (Karaletsos & Bui 2020)
- [Uncertainty estimation under model misspecification in neural network regression](https://arxiv.org/abs/2111.11763) (Cervera et. al 2021)


## Others
Hypernetwork papers that do not fall in the categories above. 
- [A Neural Representation of Sketch Drawings](https://arxiv.org/pdf/1704.03477.pdf) (Ha & Eck 2017)
- [Measuring the Intrinsic Dimension of Objective Landscapes](https://arxiv.org/abs/1804.08838) (Li et. al 2018)
- [Neural Style Transfer via Meta Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Shen_Neural_Style_Transfer_CVPR_2018_paper.html) (Shen 2018)
- [Gated Linear Networks](https://arxiv.org/abs/1910.01526) (Veness et. al 2019)
- [Hypernetwork Knowledge Graph Embeddings](https://link.springer.com/chapter/10.1007/978-3-030-30493-5_52) (Balažević et. al 2019)
- [On the Modularity of Hypernetworks](https://arxiv.org/abs/2002.10006) (Galanti & Wolf 2020)
- [Principled Weight Initialization for Hypernetworks](https://openreview.net/forum?id=H1lma24tPB) (Chang et. al 2020)

# Links to Code
The following links implemented different Hypernetwork in Pytorch code. 

- [hypnettorch](https://github.com/chrhenning/hypnettorch) (Christian Henning & Maria Cervera)
- [HyperNetworks](https://github.com/g1910/HyperNetworks) (Gaurav Mittal)

# Talks

- [Hypernetworks: a versatile and powerful tool](https://www.youtube.com/watch?v=KY9DoutzH6k) (Lior Wolf)

## License
License: MIT
