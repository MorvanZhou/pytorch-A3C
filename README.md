# Simple implementation of Reinforcement Learning (A3C) using Pytorch

This is a toy example of using multiprocessing in Python to asynchronously train a
neural network to play discrete action [CartPole](https://gym.openai.com/envs/CartPole-v0/) and
continuous action [Pendulum](https://gym.openai.com/envs/Pendulum-v0/).
The asynchronous algorithm I used is called [Asynchronous Advantage Actor-Critic](https://arxiv.org/pdf/1602.01783.pdf) or A3C.

I believe it would be the most simple toy implementation you can find at the moment (2018-01).

## What are the main focuses in this implementation?

* Pytorch + multiprocessing (NOT threading) for parallel training
* Both discrete and continuous action environments
* To be simple and easy to dig into the code (less than 200 lines)

## Reason to use [Pytorch](http://pytorch.org/) instead of [Tensorflow](https://www.tensorflow.org/)

Both of them are great for building your customized neural network. But to work
with multiprocessing, Tensorflow is not that great due to its low compatibility with multiprocessing.
I have an implementation of [Tensorflow A3C build on threading](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/10_A3C).
I even tried to implement [distributed Tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/10_A3C/A3C_distributed_tf.py).
However, the distributed version is for cluster computing which I don't have.
When using only one machine, it is slower than threading version I wrote.

Fortunately, Pytorch gets the [multiprocessing compatibility](http://pytorch.org/docs/master/notes/multiprocessing.html).
I went through many Pytorch A3C examples ([there](https://github.com/ikostrikov/pytorch-a3c), [there](https://github.com/jingweiz/pytorch-rl)
and [there](https://github.com/ShangtongZhang/DeepRL)). They are great but too complicated to dig into the code.
Therefore, this is my motivation to write my simple example codes.

## Results

![cartpole](/cartpole.png)
![pendulum](/pendulum.png)

## Dependencies

* pytorch >= 0.3
* numpy
* gym
