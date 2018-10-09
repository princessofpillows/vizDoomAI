# Description

This is a simple agent trained to perform one of three actions: shoot, move left, move right.
The goal of this scenario is to find the best Q values for quickly defeating a monster, as there is a living reward of -1, shooting and missing reward of -6, and killing the monster reward of +100.
This was built using the VIZDoom engine, which is a project created for 3D reinforcement learning research. Read more about it here: http://vizdoom.cs.put.edu.pl/.

#### Requirements

* [python3](https://realpython.com/installing-python/) to run the code
* [pip3](https://pip.pypa.io/en/latest/installing.html) to install Python dependencies
* [vizDoom](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md) to run the DOOM engine/api
* [TensorFlow](https://www.tensorflow.org/install/) for creation of model and graph
* [tqdm](https://pypi.python.org/pypi/tqdm) to view progress throughout runtime
