
import argparse
import tensorflow as tf

# ----------------------------------------
# Global variables
arg_lists = []
parser = argparse.ArgumentParser()

# Possible actions
shoot = [1, 0, 0]
left = [0, 1, 0]
right = [0, 0, 1]


# ----------------------------------------
# Macro for arparse
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# ----------------------------------------
# Arguments for preprocessing
pre_arg = add_argument_group("Preprocessing")

pre_arg.add_argument("--resolution", type=int,
                       default=(84, 84),
                       help="Resize dimensions for input")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-5,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--discount", type=float,
                       default=0.99,
                       help="Ratio of new Q value to use in update")

train_arg.add_argument("--epsilon", type=float,
                       default=1.0,
                       help="Probability of exploration")

train_arg.add_argument("--episodes", type=int,
                       default=100,
                       help="Number of episodes to train on")

train_arg.add_argument("--freq", type=int,
                       default=10,
                       help="Frequency to decrease epsilon")

train_arg.add_argument("--entropy_rate", type=int,
                       default=1e-2,
                       help="Ratio of entropy regularization to apply to loss")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs",
                       help="Directory to save logs and current model")

# ----------------------------------------
# Arguments for testing
test_arg = add_argument_group("Testing")

test_arg.add_argument("--test_episodes", type=int,
                       default=10,
                       help="Number of episodes to test on")

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--actions", type=int,
                       default=[shoot, left, right],
                       help="Possible actions to take")

model_arg.add_argument("--num_actions", type=int,
                       default=3,
                       help="Number of possible actions to take")

model_arg.add_argument("--skiprate", type=int,
                       default=3,
                       help="Number of frames to skip during each action")

model_arg.add_argument("--activ",
                       default=tf.nn.relu,
                       help="Activation function to use")

model_arg.add_argument("--init",
                       default=tf.contrib.layers.xavier_initializer(),
                       help="Initialization function to use")

# ----------------------------------------
# Function to be called externally
def get_config():
    config, unparsed = parser.parse_known_args()

    # If there are unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        parser.print_usage()
        exit(1)

    return config
