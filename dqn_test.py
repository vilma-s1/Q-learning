from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym

# Configuration paramaters for the whole setup
seed = 42

model = keras.models.load_model('/content/drive/MyDrive/game_ai/model', compile = False)

# Use the Baseline Atari environment because of Deepmind helper functions
env = make_atari("BreakoutNoFrameskip-v4")
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

env = gym.wrappers.Monitor(env, '/content/drive/MyDrive/game_ai/videos' ,
video_callable=lambda episode_id : True , force=True)

n_episodes = 10
returns = []

for _ in range(n_episodes):
  ret = 0
  state = np.array(env.reset())
  done = False

  while not done:
    # Predict action Q-values
    # From environment state
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)
    # Take best action
    action = tf.argmax(action_probs[0]).numpy()

    # Apply the sampled action in our environment
    state_next, reward, done, _ = env.step(action)
    state_next = np.array(state_next)

    ret += reward
    state = state_next
  
  returns.append(ret)

env.close()

print('Returns: _{}' . format (returns))