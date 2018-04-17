import gym
from os import path
import tensorflow as tf


TIME_STEP = 10
N_S = 10
N_A = 4
CELL_SIZE = 32
LR =  0.01


class IpaneraEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.ep_state = tf.placeholder(tf.float32, [None, N_S], 'S')
        self.ep_action = tf.placeholder(tf.float32, [None, N_A], 'A')
        self.ep_observation = tf.placeholder(tf.float32, [None, N_S], 'Observation')
        self.v, self.s_params, self.r_params = self._build_net('model')






        with tf.name_scope('step'):
            pass
        return 0

    def step(self, action):

        v_s_ = tf.Session.run(self.v, {self.ep_state: s_[np.newaxis, :], self.init_state: inputs})[0, 0]
        pass

    def reset(self):
        tf.reset_default_graph()
        return 0

    def render(self):
        return 0

    def close(self):
        return 0

    def seed(self, seed=None):
        return 0

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('statespace'):   # only critic controls the rnn update
            cell_size = 64
            s = tf.expand_dims(self.ep_state, axis=1,
                               name='timely_input')  # [time_step, feature] => [time_step, batch, feature]
            rnn_cell = tf.contrib.rnn.BasicRNNCell(cell_size)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell, inputs=s, initial_state=self.init_state, time_major=True)
            cell_out = tf.reshape(outputs, [-1, cell_size], name='flatten_rnn_outputs')  # joined state representation
            l_s = tf.layers.dense(cell_out, 50, tf.nn.relu6, kernel_initializer=w_init, name='ls')
            v = tf.layers.dense(l_s, N_S, kernel_initializer=w_init, name='v')  # state value

        with tf.variable_scope('reward'):  # state representation is based on critic
            l_r = tf.layers.dense(s, 80, tf.nn.relu6, kernel_initializer=w_init, name='la')
            r = tf.layers.dense(l_r, 1, kernel_initializer=w_init, name='v')  # state value

        s_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        r_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return v, r, s_params, r_params

