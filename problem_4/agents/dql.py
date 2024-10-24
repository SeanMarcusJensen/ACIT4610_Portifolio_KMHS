import tensorflow as tf
import keras
import numpy as np
import random
from collections import deque


class DQLAgent:
    def __init__(self, observation_space, action_space,
                 replay_buffer_size=10000, batch_size=64,
                 learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1,
                 sync_target_every=100):
        self.observation_space = observation_space
        self.action_space = action_space
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.sync_target_every = sync_target_every

        # Initialize policy and target networks
        self.policy_network = self.create_model()
        self.target_network = self.create_model()
        self.target_network.set_weights(
            self.policy_network.get_weights())  # Sync weights

        # Optimizer
        self.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate)
        self.loss_fn = keras.losses.MeanSquaredError()

        # Counter for syncing target network
        self.step_counter = 0

    def create_model(self):
        """Creates the Q-network model"""
        model = keras.Sequential([
            keras.layers.Embedding(
                self.observation_space, 128, input_length=1),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            # Q-values for each action
            keras.layers.Dense(self.action_space, activation='linear')
        ])
        return model

    def choose_action(self, state):
        """Epsilon-greedy policy for action selection"""
        if np.random.rand() < self.epsilon:
            # Random action (exploration)
            return np.random.randint(self.action_space)
        q_values = self.policy_network(np.array([state], dtype=np.float32))
        return np.argmax(q_values[0].numpy())  # Greedy action (exploitation)

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in the replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        """Train the policy network using experiences from the replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough experiences yet

        # Sample mini-batch from replay buffer
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*minibatch))

        # Convert actions to one-hot encoded
        actions_one_hot = tf.one_hot(actions, self.action_space)

        # Perform a training step
        loss = self.train_step(states, actions_one_hot,
                               rewards, next_states, dones)

        # Decrease epsilon (exploration) after each step
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Sync target network every 'sync_target_every' steps
        self.step_counter += 1
        if self.step_counter % self.sync_target_every == 0:
            self.sync_target_network()

        return loss

    @tf.function
    def train_step(self, states, actions_one_hot, rewards, next_states, dones):
        """One step of training using GradientTape"""
        # Calculate target Q-values
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)

        next_q_values = self.target_network(next_states)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        targets = rewards + self.gamma * \
            max_next_q_values * (1.0 - dones)

        with tf.GradientTape() as tape:
            # Get current Q-values for the actions taken
            q_values = self.policy_network(states)
            q_action_values = tf.reduce_sum(q_values * actions_one_hot, axis=1)

            # Compute loss
            loss = self.loss_fn(targets, q_action_values)

        # Apply gradients to update policy network
        gradients = tape.gradient(
            loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.policy_network.trainable_variables))  # type: ignore

        return loss

    def sync_target_network(self):
        """Sync target network weights with the policy network"""
        self.target_network.set_weights(self.policy_network.get_weights())

    def run_episode(self, env, max_steps):
        """Runs a single episode"""
        state = env.reset()[0]
        total_reward = 0

        for step in range(max_steps):
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            self.store_experience(state, action, reward,
                                  next_state, terminated)
            loss = self.train()  # Train after every step
            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        return total_reward


if __name__ == '__main__':
    import gym

    env = gym.make('Taxi-v3')
    observation_space = env.observation_space.n
    action_space = env.action_space.n

    agent = DQLAgent(observation_space, action_space)

    num_episodes = 1000
    max_steps_per_episode = 100

    for episode in range(num_episodes):
        reward = agent.run_episode(env, max_steps_per_episode)
        print(f"Episode {episode + 1}: Total Reward = {reward}")
