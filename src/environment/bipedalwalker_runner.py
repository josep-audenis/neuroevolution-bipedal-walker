import gymnasium as gym
import numpy as np
import random
import imageio.v2 as imageio

#import psutil
import gc
import os

from evolution.neural_network import NeuralNetwork

def evaluate_genome(genome, input_size, hidden1_size, hidden2_size, output_size, n_genome, generation, seed, render=False):
    render = render and (generation == 0 or (generation+1) % 1000 == 0)

    env = gym.make("BipedalWalker-v3", render_mode="rgb_array" if render else None)
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    done = False

    if render:
        tmp_gif_path = f"assets/gifs/bipedalwalker_{generation + 1}.{n_genome + 1}.{seed}.gif"
        writer = imageio.get_writer(tmp_gif_path, fps=45, loop=0)

    nn = NeuralNetwork(input_size, hidden1_size, hidden2_size, output_size, genome)
    n_step = 0
    while not done:

        #print(f"[{n_genome}]({n_step}) RAM used: {psutil.Process().memory_info().rss / 1e6:.2f} MB")
        n_step += 1
        if render:
            frame = env.render()
            writer.append_data(frame)
            del frame

        output = nn.forward(obs)
        action = output

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        done = terminated or truncated
        
    env.close()

    if render:
        writer.close()
        gc.collect()
        final_gif_path = f"assets/gifs/bipedalwalker_{generation + 1}.{n_genome + 1}.{seed}_{total_reward:.2f}.gif"
        os.rename(tmp_gif_path, final_gif_path)

    return total_reward

