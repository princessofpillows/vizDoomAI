import vizdoom as vzd
import tensorflow as tf
import numpy as np
import random, os, cv2
from tensorboardX import SummaryWriter
from datetime import datetime
from pathlib import Path
from tqdm import trange
from config import get_config
from models import Atari


tf.enable_eager_execution()

class replay_memory(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = []

    def push(self, exp):
        size = len(self.memory)
        # Remove oldest memory first
        if size == self.cfg.cap:
            self.memory.pop(random.randint(0, size-1))
        self.memory.append(exp)
    
    def fetch(self):
        size = len(self.memory)
        # Select batch
        if size < self.cfg.batch_size:
            batch = random.sample(self.memory, size)
        else:
            batch = random.sample(self.memory, self.cfg.batch_size)
        
        return zip(*batch)


class DQN(object):

    def __init__(self, cfg):
        super(DQN, self).__init__()

        self.game = vzd.DoomGame()
        self.game.load_config("vizdoom/scenarios/basic.cfg")
        self.game.init()

        self.cfg = cfg
        # Assign replay to CPU due to GPU memory limitations
        with tf.device('CPU:0'):
            self.replay_memory = replay_memory(cfg)
        self.global_step = tf.train.get_or_create_global_step()
        self.terminal = tf.zeros([84, 84, 1])

        # Create network
        models = {'atari': Atari}
        self.model = models[self.cfg.model](self.cfg, len(cfg.actions))
        self.model.build((None,) + self.model.shape + (cfg.num_frames,))
        self.optimizer = tf.train.AdamOptimizer(self.cfg.learning_rate)

        # Create target network
        self.target = models[self.cfg.model](self.cfg, len(cfg.actions))
        self.target.build((None,) + self.target.shape + (cfg.num_frames,))

        self.update_target()
        self.build_writers()

    def build_writers(self):
        if not Path(cfg.save_dir).is_dir():
            os.mkdir(cfg.save_dir)
        if cfg.extension is None:
            cfg.extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.save_path = cfg.save_dir + cfg.extension
        self.ckpt_prefix = self.save_path + '/ckpt'
        self.saver = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, optimizer_step=self.global_step)

        log_path = cfg.log_dir + cfg.extension
        self.writer = SummaryWriter(log_path)

    def logger(self, tape, loss):
        if self.global_step.numpy() % cfg.log_freq == 0:
            # Log scalars
            self.writer.add_scalar('loss', loss.numpy(), self.global_step)
            
            # Log weight scalars
            slots = self.optimizer.get_slot_names()
            for variable in tape.watched_variables():
                    self.writer.add_scalar(variable.name, tf.nn.l2_loss(variable).numpy(), self.global_step)

                    for slot in slots:
                        slotvar = self.optimizer.get_slot(variable, slot)
                        if slotvar is not None:
                            self.writer.add_scalar(variable.name + '/' + slot, tf.nn.l2_loss(slotvar).numpy(), self.global_step)

    def log_state(self, frames):
        # Log state
        for i in range(len(frames)):
            s = np.reshape(frames[i], [1, frames[i].shape[0], frames[i].shape[1]]).astype(np.uint8)
            self.writer.add_image('state ' + 'n-' + str(i), s, self.global_step)

    def update_target(self):
        self.target.set_weights(self.model.get_weights())

    def update(self):
        # Fetch batch of experiences
        s0, logits, rewards, s1 = self.replay_memory.fetch()
        
        # Get entropy
        probs = tf.nn.softmax(logits)
        entropy = -1 * tf.reduce_sum(probs*tf.math.log(probs + 1e-20))
        # Construct graph
        with tf.GradientTape() as tape:
            # Get predicted q values
            logits, _ = self.model(s0)
            # Choose max q values for all batches
            rows = tf.range(tf.shape(logits)[0])
            cols = tf.argmax(logits, 1, output_type=tf.int32)
            rows_cols = tf.stack([rows, cols], axis=1)
            q = tf.gather_nd(logits, rows_cols)

            # Get target q values
            target_logits, _ = self.target(s1)
            rows = tf.range(tf.shape(target_logits)[0])
            # Using columns of selected actions from prediction, stack with target rows
            rows_cols = tf.stack([rows, cols], axis=1)
            # Slice Q values, with actions chosen by prediction
            target_q = tf.gather_nd(logits, rows_cols)
            # Kill target network gradient
            target_q = tf.stop_gradient(target_q)

            # Compare target q and predicted q (q = 0 on terminal state)
            loss = 1/2 * tf.reduce_mean(tf.losses.huber_loss(rewards + self.cfg.discount * target_q, q)) - entropy * self.cfg.entropy_rate
        
        self.logger(tape, loss)
        # Compute/apply gradients
        grads = tape.gradient(loss, self.model.trainable_weights)
        grads_and_vars = zip(grads, self.model.trainable_weights)
        self.optimizer.apply_gradients(grads_and_vars)

        self.global_step.assign_add(1)
    
    def e_greedy(self, choice):
        if random.random() > self.cfg.epsilon:
            return choice.tolist()
        else:
            return random.choice(self.cfg.actions)

    def perform_action(self, frames):
        logits, _ = self.model(frames)
        # Get indexes of highest Q values
        cols = tf.argmax(logits, 1, output_type=tf.int32)
        # Slice highest Q values
        q = logits 
        # One-hot action
        choice = np.zeros(len(self.cfg.actions))
        choice[tuple(cols)] += 1
        # Take action
        reward = self.game.make_action(self.e_greedy(choice), self.cfg.skiprate)
        # Modify rewards (game is blackbox)
        '''
        if reward > 50:
            reward = 10
        elif reward < -6:
            reward = -3
        else:
            reward = -1
        '''
        return reward, logits, q
    
    def preprocess(self):
        screen = self.game.get_state().screen_buffer
        frame = np.multiply(screen, 255.0/screen.max())
        frame = tf.image.rgb_to_grayscale(frame)
        frame = tf.image.resize_images(frame, self.model.shape)
        return frame
    
    def train(self):
        self.saver.restore(tf.train.latest_checkpoint(self.cfg.save_dir))
        for episode in trange(self.cfg.episodes):
            # Reduce exploration rate linearly. Capped at 0.1 halfway through training
            if self.cfg.epsilon > 0.1:
                self.cfg.epsilon -= 0.9 / (0.5 * self.cfg.episodes)

            # Save model
            if episode % self.cfg.save_freq == 0:
                self.saver.save(file_prefix=self.ckpt_prefix)

            # Setup variables
            self.game.new_episode()
            frame = self.preprocess()
            # Init stack of 4 frames
            frames = [frame, frame, frame, frame]

            count = 0
            while not self.game.is_episode_finished():
                s = tf.reshape(frames, [1, self.model.shape[0], self.model.shape[1], self.cfg.num_frames])
                reward, logits, q = self.perform_action(s)

                if count % self.cfg.num_frames == 0:
                    self.log_state(frames)

                # Update frames with latest image
                prev_frames = frames[:]
                frames.pop(0)
                # Reached terminal state, kill q values
                if self.game.get_state() is None:
                    frames = [self.terminal, self.terminal, self.terminal, self.terminal]
                    logits = tf.zeros(logits.shape)
                else:
                    frames.append(self.preprocess())

                s0 = tf.reshape(prev_frames, [self.model.shape[0], self.model.shape[1], self.cfg.num_frames])
                s1 = tf.reshape(frames, [self.model.shape[0], self.model.shape[1], self.cfg.num_frames])
                #priority = tf.abs(reward + self.cfg.discount * q - self.prev_q) + 1e-20
                # Populate memory with experiences
                self.replay_memory.push([s0, logits, reward, s1])

                count += 1
                
            # Train on experiences from memory
            self.update()
            # Update target network
            if episode % (self.cfg.update_target_rate * self.cfg.episodes) == 0:
                self.update_target()
        
        self.saver.save(file_prefix=self.ckpt_prefix)

    def test(self):
        self.saver.restore(tf.train.latest_checkpoint(self.cfg.save_dir))
        rewards = []
        for _ in trange(self.cfg.test_episodes):
            # Setup variables
            self.game.new_episode()
            frame = self.preprocess()
            # Init stack of 4 frames
            frames = [frame, frame, frame, frame]

            while not self.game.is_episode_finished():
                s = tf.reshape(frames, [1, self.model.shape[0], self.model.shape[1], self.cfg.num_frames])
                _, _, _ = self.perform_action(s)
                
                # Update frames with latest image
                if self.game.get_state() is not None:
                    frames.pop(0)
                    frames.append(self.preprocess())

            rewards.append(self.game.get_total_reward())
        print("Average Reward: ", sum(rewards)/self.cfg.test_episodes)

def main(cfg):
    model = DQN(cfg)

    model.train()

    model.test()

if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
