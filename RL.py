#!/usr/bin/env python3
"""
Backtesting and RL trading system for XAUUSD with rolling window predictions.
"""

import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def load_config() -> Dict[str, Any]:
    """Return configuration parameters for RL and backtesting."""
    config = {
        # Data files
        'data_csv': 'xauusd_1min.csv',
        'predictions_csv': 'xauusd_predictions.csv',
        # Data processing parameters
        'rolling_window': 1440,  # number of samples in rolling window (1 day)
        'prediction_horizon': 90,  # number of future samples to predict
        # Trading simulation parameters
        'initial_capital': 1000.0,  # initial money in dollars
        'transaction_cost': 0.0,  # transaction cost (as a fraction)
        'lot_sizes': {  # lot sizes for trades
            'small': 0.01,
            'medium': 0.05,
            'large': 0.1
        },
        # RL parameters
        'rl': {
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 10,  # number of episodes (use a higher number for production)
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.995,
            'model_checkpoint_path': 'rl_checkpoint.h5',
            'visualization': True,  # set to False to disable training visualization
            'update_target_every': 1000,
            'memory_size': 10000
        }
    }
    return config


def load_data(file_path: str) -> pd.Series:
    """Load historical close price data from CSV file."""
    df = pd.read_csv(file_path)
    # Assumes CSV contains at least a 'close' column.
    return df['close']


def predict_future(window: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Placeholder for future prediction using a deep model.

    In production, replace this with a call to your external deep model.
    """
    prediction_horizon = config['prediction_horizon']
    last_value = window[-1]
    # Dummy prediction: repeat last value with some added noise.
    return last_value + np.random.randn(prediction_horizon) * 0.01


def generate_predictions(data: pd.Series, config: Dict[str, Any]) -> pd.DataFrame:
    """Generate future predictions using a rolling window and prediction model."""
    rolling_window = config['rolling_window']
    prediction_horizon = config['prediction_horizon']
    predictions = []

    # Loop over all valid windows (starting from index 'rolling_window' up to len(data)-prediction_horizon)
    for i in range(rolling_window, len(data) - prediction_horizon + 1):
        window = data.iloc[i - rolling_window: i].values
        pred = predict_future(window, config)
        predictions.append(pred)
    # Create a DataFrame with columns for each future sample and record the corresponding index.
    columns = [f'pred_{j}' for j in range(prediction_horizon)]
    pred_df = pd.DataFrame(predictions, columns=columns)
    # Save the original data index where each prediction applies.
    pred_df['index'] = range(rolling_window, len(data) - prediction_horizon + 1)
    return pred_df


def save_predictions(predictions: pd.DataFrame, file_path: str) -> None:
    """Save predictions DataFrame to a CSV file."""
    predictions.to_csv(file_path, index=False)


def load_predictions(file_path: str) -> pd.DataFrame:
    """Load predictions from CSV file."""
    return pd.read_csv(file_path)


def get_state(index: int, data: pd.Series, predictions: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
    """Get state vector from current price and future predictions."""
    current_price = data.iloc[index]
    # Retrieve the row in predictions corresponding to the current index.
    row = predictions[predictions['index'] == index]
    if row.empty:
        # If no prediction exists, use a placeholder (e.g. current price repeated).
        pred_values = np.full((config['prediction_horizon'],), current_price)
    else:
        pred_values = row.drop('index', axis=1).values.flatten()
    # State: current price concatenated with the prediction vector.
    state_vec = np.concatenate(([current_price], pred_values))
    return state_vec


def env_reset(data: pd.Series, predictions: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray]:
    """Reset environment state and return initial state dictionary and state vector."""
    initial_index = config['rolling_window']
    state = {'index': initial_index, 'capital': config['initial_capital']}
    state_vec = get_state(initial_index, data, predictions, config)
    return state, state_vec


def env_step(state: Dict[str, Any], action: int, data: pd.Series, predictions: pd.DataFrame,
             config: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray, float, bool]:
    """
    Perform one environment step based on the action.

    Returns new state dict, new state vector, reward, and done flag.
    """
    current_index = state['index']
    current_capital = state['capital']
    prediction_horizon = config['prediction_horizon']
    transaction_cost_rate = config.get('transaction_cost', 0.0)
    lot_sizes = config.get('lot_sizes', {'small': 0.01, 'medium': 0.05, 'large': 0.1})

    # Map actions to trade decisions:
    # 0: hold, 1: buy small, 2: buy medium, 3: buy large, 4: sell small, 5: sell medium, 6: sell large.
    if action == 0:
        trade_type = 'hold'
        lot_size = 0.0
    elif action == 1:
        trade_type = 'buy'
        lot_size = lot_sizes['small']
    elif action == 2:
        trade_type = 'buy'
        lot_size = lot_sizes['medium']
    elif action == 3:
        trade_type = 'buy'
        lot_size = lot_sizes['large']
    elif action == 4:
        trade_type = 'sell'
        lot_size = lot_sizes['small']
    elif action == 5:
        trade_type = 'sell'
        lot_size = lot_sizes['medium']
    elif action == 6:
        trade_type = 'sell'
        lot_size = lot_sizes['large']
    else:
        trade_type = 'hold'
        lot_size = 0.0

    # Determine the future index at which the trade will be "closed"
    future_index = current_index + prediction_horizon
    if future_index >= len(data):
        # End of available data: mark as done.
        done = True
        reward = 0.0
        new_index = current_index
        new_state = {'index': new_index, 'capital': current_capital}
        new_state_vec = get_state(new_index, data, predictions, config)
        return new_state, new_state_vec, reward, done

    entry_price = data.iloc[current_index]
    exit_price = data.iloc[future_index]

    # Compute profit/loss based on trade type.
    if trade_type == 'buy':
        profit = (exit_price - entry_price) * lot_size
    elif trade_type == 'sell':
        profit = (entry_price - exit_price) * lot_size
    else:
        profit = 0.0

    # Deduct transaction cost.
    transaction_cost = transaction_cost_rate * entry_price * lot_size
    reward = profit - transaction_cost
    new_capital = current_capital + reward
    new_index = current_index + 1
    done = new_index >= (len(data) - prediction_horizon)
    new_state = {'index': new_index, 'capital': new_capital}
    new_state_vec = get_state(new_index, data, predictions, config)
    return new_state, new_state_vec, reward, done


def build_q_network(input_shape: Tuple[int, ...], num_actions: int,
                    config: Dict[str, Any]) -> tf.keras.Model:
    """Build and return the Q-network model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_actions)
    ])
    return model


def choose_action(state: np.ndarray, model: tf.keras.Model, epsilon: float) -> int:
    """Choose an action using an epsilon-greedy policy."""
    if np.random.rand() < epsilon:
        return np.random.randint(0, 7)
    q_values = model(np.expand_dims(state, axis=0), training=False)
    return int(np.argmax(q_values.numpy()))


def update_q_network(model: tf.keras.Model, target_model: tf.keras.Model,
                     optimizer: tf.keras.optimizers.Optimizer, batch: List[Tuple[Any, ...]],
                     config: Dict[str, Any]) -> float:
    """Update the Q-network using a batch of transitions and return the loss."""
    gamma = config['rl']['gamma']
    states = np.array([transition[0] for transition in batch])
    actions = np.array([transition[1] for transition in batch])
    rewards = np.array([transition[2] for transition in batch], dtype=np.float32)
    next_states = np.array([transition[3] for transition in batch])
    dones = np.array([transition[4] for transition in batch], dtype=np.float32)

    with tf.GradientTape() as tape:
        q_values = model(states, training=True)
        # Gather Q-values for chosen actions.
        q_values = tf.reduce_sum(q_values * tf.one_hot(actions, q_values.shape[1]), axis=1)
        next_q_values = target_model(next_states, training=False)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        targets = rewards + (1 - dones) * gamma * max_next_q_values
        loss = tf.reduce_mean(tf.square(targets - q_values))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss.numpy()


def train_rl_agent(data: pd.Series, predictions: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Train the RL agent using DQN on the trading simulation environment."""
    num_episodes = config['rl']['epochs']
    epsilon = config['rl']['epsilon_start']
    epsilon_min = config['rl']['epsilon_min']
    epsilon_decay = config['rl']['epsilon_decay']
    batch_size = config['rl']['batch_size']
    memory: List[Tuple[Any, ...]] = []
    memory_max_size = config['rl']['memory_size']
    update_target_every = config['rl'].get('update_target_every', 1000)
    state_size = 1 + config['prediction_horizon']  # current price + predictions
    num_actions = 7  # defined actions: hold, buy/sell with three lot sizes

    model = build_q_network((state_size,), num_actions, config)
    target_model = build_q_network((state_size,), num_actions, config)
    target_model.set_weights(model.get_weights())
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['rl']['learning_rate'])

    total_steps = 0
    rewards_per_episode = []

    for episode in range(num_episodes):
        state_dict, state_vec = env_reset(data, predictions, config)
        episode_reward = 0.0
        done = False
        while not done:
            action = choose_action(state_vec, model, epsilon)
            new_state_dict, new_state_vec, reward, done = env_step(state_dict, action, data, predictions, config)
            episode_reward += reward

            # Store transition in replay memory.
            memory.append((state_vec, action, reward, new_state_vec, done))
            if len(memory) > memory_max_size:
                memory.pop(0)

            state_vec = new_state_vec
            state_dict = new_state_dict
            total_steps += 1

            # Update Q-network if there is enough experience.
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                loss = update_q_network(model, target_model, optimizer, batch, config)

            # Periodically update the target network.
            if total_steps % update_target_every == 0:
                target_model.set_weights(model.get_weights())

        rewards_per_episode.append(episode_reward)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

    # Save the trained model.
    model.save(config['rl']['model_checkpoint_path'])
    print(f"RL model saved to {config['rl']['model_checkpoint_path']}.")

    # Optional visualization of training rewards.
    if config['rl'].get('visualization', False):
        plt.plot(rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.show()


def main() -> None:
    """Main function to run prediction generation and RL training."""
    config = load_config()
    # Load historical data (CSV must have a 'close' column).
    data = load_data(config['data_csv'])

    # Generate predictions if the predictions file does not exist.
    if not os.path.exists(config['predictions_csv']):
        print("Generating predictions from historical data...")
        predictions_df = generate_predictions(data, config)
        save_predictions(predictions_df, config['predictions_csv'])
        print(f"Predictions saved to {config['predictions_csv']}.")
    else:
        print("Loading existing predictions...")
        predictions_df = load_predictions(config['predictions_csv'])

    # Train the RL agent.
    train_rl_agent(data, predictions_df, config)


if __name__ == '__main__':
    main()
