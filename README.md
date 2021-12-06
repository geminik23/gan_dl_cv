# Unsupervised Learning by Generative Adversarial Nets

## Aim
1. obtain practical knowledge and hands-on understanding of the basic concepts in Generative Adversarial Nets(GAN)
2. obtain practical experience on how to implement basic GAN using tensorflow.

## Understanding GAN models basic concepts

Objective: To become familiar with basic of GAN model and its basic usages.

Reference: https://towardsdatascience.com/understanding-generative-adversarialnetworks-4dafc963f2ef

## Generative Adversarial Networks with TensorFlow
Objective: To become familiar with GAN and re-implement the original GAN model

1. the architecture of discriminator and generator as follow:

![architecture](./structure.png)

2. remove dropout function for this architecture, and observe its training 
convergence.

3. show the generated images at 10 epochs, 20 epochs,50 epochs,100 
epochs


## Report
[A Critical Analysisof ‘Generative Adversarial Nets’](./report.pdf)


## Result
Epoch 10
![epoch_10](./results/MNIST_GAN_10.png)

Epoch 20
![epoch_20](./results/MNIST_GAN_20.png)

Epoch 50
![epoch_50](./results/MNIST_GAN_50.png)

Epoch 100
![epoch_100](./results/MNIST_GAN_100.png)

Train Histograms
![histogram](./MNIST_GAN_train_hist.png)
