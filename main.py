import vizdoom as vzd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random, cProfile, pstats
from tqdm import trange
from config import get_config

tf.enable_eager_execution()

class Q_Learning(object):

    def __init__(self, cfg):
        super(Q_Learning, self).__init__()

        self.cfg = cfg

        self.game = vzd.DoomGame()
        self.game.load_config("vizdoom/scenarios/basic.cfg")
        self.game.init()

        self.model = tf.keras.Sequential([
            # Filters, Kernel Size, Strides
            tf.keras.layers.Conv2D(32, 8, 4, activation=self.cfg.activ, kernel_initializer=self.cfg.init),
            tf.keras.layers.Conv2D(64, 4, 2, padding="same", activation=self.cfg.activ, kernel_initializer=self.cfg.init),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation=self.cfg.activ, kernel_initializer=self.cfg.init),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=self.cfg.activ, kernel_initializer=self.cfg.init),
            tf.keras.layers.Dense(self.cfg.num_actions, kernel_initializer=self.cfg.init)
        ])
        # Specify input size
        self.model.build((4, 84, 84, 1))
        self.optimizer = tf.train.AdamOptimizer(self.cfg.learning_rate)
    
    def preprocess(self):
        state = self.game.get_state()
        screen = tf.image.resize_images(state.screen_buffer, self.cfg.resolution)
        screen = tf.image.rgb_to_grayscale(screen)
        self.frames.append(screen)

    def update(self):
        # Perform forward pass, construct graph
        with tf.GradientTape() as tape:
            self.forward()
            reward = self.game.make_action(self.e_greedy())
            loss = tf.losses.mean_squared_error(reward + self.cfg.discount * self.new_q, self.old_q)

        # Compute/apply gradients
        grads = tape.gradient(loss, self.model.weights)
        grads_and_vars = zip(grads, self.model.weights)
        self.optimizer.apply_gradients(grads_and_vars)

    def e_greedy(self):
        if random.random() > self.cfg.epsilon:
            return self.choice
        else:
            return random.choice(self.cfg.actions)

    def forward(self):
        # Get Q values for all actions
        logits = self.model(self.frames)[0]
        # Get highest Q value index
        idx = tf.argmax(logits, 0)
        self.new_q = logits[idx]
        # Update action to take
        self.choice = [0, 0, 0]
        self.choice[idx] += 1
    
    def train(self):
        self.old_q = 0
        self.frames = []
        rewards = []
        for i in trange(self.cfg.episodes):
            # Reduce exploration rate
            if i % self.cfg.freq == 0:
                self.cfg.epsilon -= 0.1

            self.game.new_episode()
            while not self.game.is_episode_finished():
                self.preprocess()
                # Ensure batch size is 4
                if len(self.frames) == 4:
                    self.update()
                    # Remove oldest frame
                    self.frames.pop(0)
            rewards.append(self.game.get_total_reward())

        plt.plot(rewards)
        plt.show()

    def test(self):
        self.old_q = 0
        self.frames = []
        rewards = []
        for i in trange(self.cfg.test_episodes):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                self.preprocess()
                # Ensure batch size is 4
                if len(self.frames) == 4:
                    self.update()
                    # Remove oldest frame
                    self.frames.pop(0)
            rewards.append(self.game.get_total_reward())

        print(np.asarray(rewards)/self.cfg.test_episodes)

def main(cfg):
    model = Q_Learning(cfg)

    model.train()

    model.test()

if __name__ == "__main__":
    cfg = get_config()

    cProfile.run('main(cfg)', 'prof')
    p = pstats.Stats('prof')
    p.strip_dirs().sort_stats('cumulative').print_stats(50)
