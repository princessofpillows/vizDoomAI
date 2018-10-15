import vizdoom as vzd
import tensorflow as tf
import numpy as np
import random, cProfile, pstats
from tqdm import trange
from config import get_config
from utils.diagnostic import diagnostic


tf.enable_eager_execution()

class Q_Learning(object):

    def __init__(self, cfg):
        super(Q_Learning, self).__init__()

        self.cfg = cfg

        self.game = vzd.DoomGame()
        self.game.load_config("vizdoom/scenarios/basic.cfg")
        self.game.init()

        self.global_step = tf.train.get_or_create_global_step()
        self.writer = tf.contrib.summary.create_file_writer(self.cfg.log_dir)
        self.writer.set_as_default()

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
        screen = self.game.get_state().screen_buffer
        screen = np.multiply(screen, 255.0/screen.max())
        #screen = tf.image.per_image_standardization(state.screen_buffer)
        screen = tf.image.rgb_to_grayscale(screen)
        screen = tf.image.resize_images(screen, self.cfg.resolution)
        self.frames.append(screen)

    def logger(self):
        with tf.contrib.summary.record_summaries_every_n_global_steps(10, self.global_step):
            # Log variables
            tf.contrib.summary.scalar('loss', self.loss)
            tf.contrib.summary.scalar('reward', self.reward)
            tf.contrib.summary.scalar('action', self.idx)
            tf.contrib.summary.scalar('q-values', self.new_q)

            # Log weights
            slots = self.optimizer.get_slot_names()
            for variable in self.tape.watched_variables():
                    tf.contrib.summary.scalar(variable.name, tf.nn.l2_loss(variable))

                    for slot in slots:
                        slotvar = self.optimizer.get_slot(variable, slot)
                        if slotvar is not None:
                            tf.contrib.summary.scalar(variable.name + '/' + slot, tf.nn.l2_loss(slotvar))

    def update(self):
        # Perform forward pass, construct graph
        with tf.GradientTape() as self.tape:
            self.forward()
            self.reward = self.game.make_action(self.e_greedy(), self.cfg.skiprate)
            self.loss = tf.losses.mean_squared_error(self.reward + self.cfg.discount * self.new_q, self.old_q)

        self.logger()
        # Compute/apply gradients
        grads = self.tape.gradient(self.loss, self.model.weights)
        grads_and_vars = zip(grads, self.model.weights)
        self.optimizer.apply_gradients(grads_and_vars)

        self.global_step.assign_add(1)

    def e_greedy(self):
        if random.random() > self.cfg.epsilon:
            return self.choice
        else:
            return random.choice(self.cfg.actions)

    def forward(self):
        # Get Q values for all actions
        self.logits = self.model(self.frames)[0]
        # Get highest Q value index
        self.idx = tf.argmax(self.logits, 0)
        self.new_q = self.logits[self.idx]
        # Update action to take
        self.choice = [0, 0, 0]
        self.choice[self.idx] += 1
    
    def train(self):
        self.old_q = 0
        self.frames = []
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

                    #diagnostic(self.logits.numpy(), self.idx.numpy(), self.reward)
                    # Remove oldest frame
                    self.frames.pop(0)

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
                    self.forward()
                    reward = self.game.make_action(self.e_greedy(), self.cfg.skiprate)
                    rewards.append(reward)
                    # Remove oldest frame
                    self.frames.pop(0)
            rewards.append(self.game.get_total_reward())

        print("Average Reward: ", sum(rewards)/self.cfg.test_episodes)

def main(cfg):
    model = Q_Learning(cfg)

    model.train()

    model.test()

if __name__ == "__main__":
    cfg = get_config()

    cProfile.run('main(cfg)', 'prof')
    p = pstats.Stats('prof')
    p.strip_dirs().sort_stats('cumulative').print_stats(50)
