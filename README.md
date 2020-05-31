## Experiments that compare Actor-Critic algorithms such as A2C and A3C. A thesis conducted by the TU Berlin.

This is an example of using multiprocessing with Pycharm to synchronously and asynchronously train a
neural network to play 2D discrete action [CartPole](https://gym.openai.com/envs/CartPole-v0/) and
3D realistic environment [Vizdoom](https://github.com/mwydmuch/ViZDoom) games. The baseline of the Advantage Actor-Critic is A2C. The asynchronous variation I used is called [Asynchronous Advantage Actor-Critic](https://arxiv.org/pdf/1602.01783.pdf) or A3C. The synchronous version is called [Synchronous Advantage Actor-Critic](https://openai.com/blog/baselines-acktr-a2c/) or A2C-Sync.

## What are the main focuses in this implementation?

* Pytorch + multiprocessing + shared memory for parallel training
* Simple discrete environment and 3D realistic environment
* Partly shared or separate Actor-Critic Neural Networks

## Reason of using [Pytorch](http://pytorch.org/) instead of [Tensorflow](https://www.tensorflow.org/)

Both of them are great for building your customized neural network. But to work
with multiprocessing, Tensorflow is not that great due to its low compatibility with multiprocessing.
I have an implementation of [Tensorflow A3C build on threading](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/10_A3C).
I even tried to implement [distributed Tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/10_A3C/A3C_distributed_tf.py).
However, the distributed version is for cluster computing which I don't have.
When using only one machine, it is slower than threading version I wrote.

Fortunately, Pytorch gets the [multiprocessing compatibility](http://pytorch.org/docs/master/notes/multiprocessing.html).

## Codes & Results

* [shared_adam.py](/CARTPOLE/shared_adam.py): optimizer that shares its parameters in parallel
* [utils.py](/CARTPOLE/cart_utils.py): contains plots, probability distribution, optimizer and memory functions


CartPole results
![cartpole](/CARTPOLE/cart_results/Compared_results/All.png)
![cartpole_test](/CARTPOLE/cart_results/Compared_results/All_test.png)


Vizdoom results
![vizdoom](/VIZDOOM/doom_results/Compared/all.png)
![vizdoom_test](/VIZDOOM/doom_results/Compared/all_test.png)

## Dependencies

* pytorch >= 0.4.0
* numpy
* gym
* matplotlib
