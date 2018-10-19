import vizdoom as vzd
import tensorflow as tf
import numpy as np
import random, cProfile, pstats
from tqdm import trange
from config import get_config


tf.enable_eager_execution()

class replay_memory(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = []

    def push(self, exp):
        # Remove oldest memory first
        if len(self.memory) == self.cfg.cap:
            self.memory.pop(random.randint(0, len(self.memory)))
        self.memory.append(exp)
    
    def fetch(self):
        # Select batch
        if len(self.memory) < self.cfg.batch_size:
            batch = random.sample(self.memory, len(self.memory))
        else:
            batch = random.sample(self.memory, self.cfg.batch_size)
        # Return batch
        batch = np.asarray(batch, dtype=object)
        return zip(*batch)


class Q_Learning(object):

    def __init__(self, cfg):
        super(Q_Learning, self).__init__()

        self.cfg = cfg
        self.replay_memory = replay_memory(cfg)

        self.game = vzd.DoomGame()
        self.game.load_config("vizdoom/scenarios/basic.cfg")
        self.game.init()
        self.terminal = tf.zeros([84, 84, 1])

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
        self.model.build((None, 84, 84, 4))
        self.optimizer = tf.train.AdamOptimizer(self.cfg.learning_rate)

    def logger(self, tape):
        with tf.contrib.summary.record_summaries_every_n_global_steps(10, self.global_step):

            # Log weights
            slots = self.optimizer.get_slot_names()
            for variable in tape.watched_variables():
                    tf.contrib.summary.scalar(variable.name, tf.nn.l2_loss(variable))

                    for slot in slots:
                        slotvar = self.optimizer.get_slot(variable, slot)
                        if slotvar is not None:
                            tf.contrib.summary.scalar(variable.name + '/' + slot, tf.nn.l2_loss(slotvar))

    def update(self):
        # Fetch batch of experiences
        prev_frames, logits, frames, rewards = self.replay_memory.fetch()
        # Get target Q
        q = self.forward(frames, True)
        q = tf.stop_gradient(q)
        # Get entropy
        probs = tf.nn.softmax(logits)
        entropy = -1 * tf.reduce_sum(probs*tf.math.log(probs + 1e-20))
        # Construct graph
        with tf.GradientTape() as tape:
            prev_q = self.forward(prev_frames, True)
            # Compute loss (q = 0 on terminal state)
            loss = tf.losses.mean_squared_error(rewards + self.cfg.discount * q, prev_q) - entropy * self.cfg.entropy_rate
        
        self.logger(tape)
        # Compute/apply gradients
        grads = tape.gradient(loss, self.model.weights)
        grads_and_vars = zip(grads, self.model.weights)
        self.optimizer.apply_gradients(grads_and_vars)

        self.global_step.assign_add(1)

    def e_greedy(self, choice):
        if random.random() > self.cfg.epsilon:
            return choice.tolist()
        else:
            return random.choice(self.cfg.actions)

    def forward(self, frames, is_replay):
        # Get Q values for all actions
        logits = self.model(frames)
        # Get list of rows
        rows = tf.range(tf.shape(logits)[0])
        # Get indexes of highest Q values
        cols = tf.argmax(logits, 1, output_type=tf.int32)
        # Stack rows and columns
        rows_cols = tf.stack([rows, cols], axis=1)
        # Slice highest Q values
        q = tf.gather_nd(logits, rows_cols)
        if is_replay:
            return q
        else:
            # One-hot action
            choice = np.zeros(self.cfg.num_actions)
            choice[cols] += 1
            # Take action
            reward = self.game.make_action(self.e_greedy(choice), self.cfg.skiprate)
            return q, logits, reward
    
    def preprocess(self):
        screen = self.game.get_state().screen_buffer
        screen = np.multiply(screen, 255.0/screen.max())
        screen = tf.image.rgb_to_grayscale(screen)
        screen = tf.image.resize_images(screen, (84, 84))
        return screen
    
    def train(self):
        for episode in trange(self.cfg.episodes):
            # Reduce exploration rate
            if episode % self.cfg.freq == 0:
                self.cfg.epsilon -= 0.1

            # Setup variables
            self.game.new_episode()
            screen = self.preprocess()
            frames = []
            # Init stack of 4 frames
            for _ in range(4):
                frames.append(screen)

            while not self.game.is_episode_finished():
                _, logits, reward = self.forward(tf.reshape(frames, [1, 84, 84, 4]), False)
                # Update frames with latest image
                prev_frames = frames[:]
                frames.pop(0)

                # Reached terminal state
                if self.game.get_state() is None:
                    frames.append(self.terminal)
                else:
                    frames.append(self.preprocess())

                # Populate memory with experiences
                self.replay_memory.push([tf.reshape(prev_frames, [84, 84, 4]), logits, tf.reshape(frames, [84, 84, 4]), reward])
                # Train on experiences from memory
                self.update()

    def test(self):
        rewards = []
        for _ in trange(self.cfg.test_episodes):
            # Setup variables
            self.game.new_episode()
            screen = self.preprocess()
            frames = []
            # Init stack of 4 frames
            for _ in range(4):
                frames.append(screen)

            while not self.game.is_episode_finished():
                _, _, _ = self.forward(tf.reshape(frames, [1, 84, 84, 4]), False)
                # Update frames with latest image
                frames.pop(0)
                if self.game.get_state() is not None:
                    frames.append(self.preprocess())

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
