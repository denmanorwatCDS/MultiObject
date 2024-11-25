from MultiObjectEnv.MultiObjectEnv import MultipleFetchPickAndPlaceEnv

import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np
import cv2
from gym.envs.robotics import rotations
        

def throw_block(env, current_obs, video_size = 800):
    movie = []
    # Raise hand
    x, y, z, goal_x, goal_y, goal_z = current_obs['observation'][:6]
    while z - goal_z < 0.2:
        obs, _, _, _ = env.step(np.array([0., 0., 1., 0.]))
        x, y, z, goal_x, goal_y, goal_z = obs['observation'][:6]
        movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
    
    # Find to object
    while abs(x - goal_x) > 3e-03 or abs(y - goal_y) > 3e-03:
        delta_x, delta_y = 20*(goal_x - x), 20*(goal_y - y)
        total_delta = np.clip(np.array([delta_x, delta_y, 0., 0.]), -1., 1.)
        obs, _, _, _ = env.step(total_delta)
        x, y, z, goal_x, goal_y, goal_z = obs['observation'][:6]
        movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
    
    # Open gripper, place it onto object
    open_gripper = np.array([0., 0., 0., 1.])
    env.step(open_gripper)
    while abs(z - goal_z) > 1e-02:
        total_delta = np.clip(np.array([0., 0., 20*(goal_z - z), 0.]), -1., 1.)
        obs, _, _, _ = env.step(total_delta)
        x, y, z, goal_x, goal_y, goal_z = obs['observation'][:6]
        movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
    
    # Grasp object, raise it up
    for i in range(5):
        env.step(np.array([0., 0., 0., -1]))
        movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))

    while z < 0.6:
        obs, _, _, _ = env.step(np.array([0., 0., 1., -1.]))
        x, y, z, goal_x, goal_y, goal_z = obs['observation'][:6]
        movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))

    # Move grasped object
    for i in range(4):
        env.step(np.array([1., 0., 0., -1.]))
        movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
    
    # Drop object
    env.step(np.array([1., 0., 0., 1.]))
    movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
    
    # Wait for object to fall
    for i in range(25):
        env.step(np.array([0., 0., 0., 0.]))
        movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))

    return movie

def push_block(env, current_obs, video_size = 800):
    movie = []
    # Close gripper
    env.step(np.array([0., 0., 0., -1.]))
    movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
    
    # Lower hand to reach object
    x, y, z, goal_x, goal_y, goal_z = current_obs['observation'][:6]
    while abs(z - goal_z) > 1e-02:
        total_delta = np.clip(np.array([0., 0., 20*(goal_z - z), 0.]), -1., 1.)
        obs, _, _, _ = env.step(total_delta)
        x, y, z, goal_x, goal_y, goal_z = obs['observation'][:6]
        movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
        cv2.imwrite('test.png', movie[-1])

    # Move to an object, save direction
    direction = 5*(goal_x - x), 5*(goal_y - y)
    while abs(x - goal_x) > 0.1 or abs(y - goal_y) > 0.1:
        delta_x, delta_y = 20*(goal_x - x), 20*(goal_y - y)
        total_delta = np.clip(np.array([delta_x, delta_y, 0., 0.]), -1., 1.)
        obs, _, _, _ = env.step(total_delta)
        x, y, z, goal_x, goal_y, goal_z = obs['observation'][:6]
        movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
    
    # Push block
    for i in range(10):
        env.step(np.clip(np.array([direction[0], direction[1], 0., 0.]), -1., 1.))
        movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
    
    # Save for checking in impulse is passed
    for i in range(25):
        env.step(np.clip(np.array([0, 0, 0., 0.]), -1., 1.))
        movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
    return movie

def pick_and_place(env, current_obs, video_size = 800):
    movie = []
    manipulator_idx = np.array([0, 1, 2])
    object_idxs = [np.array([3, 4, 5]), np.array([6, 7, 8]), np.array([9, 10, 11]), np.array([12, 13, 14])]
    for object_idx in object_idxs:
        x, y, z = current_obs['observation'][manipulator_idx]
        goal_x, goal_y, goal_z = current_obs['observation'][object_idx]
        while z - goal_z < 0.2:
            obs, _, _, _ = env.step(np.array([0., 0., 1., 0.]))
            x, y, z = obs['observation'][manipulator_idx]
            goal_x, goal_y, goal_z = obs['observation'][object_idx]
            movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
    
        # Find to object
        while abs(x - goal_x) > 3e-03 or abs(y - goal_y) > 3e-03:
            delta_x, delta_y = 20*(goal_x - x), 20*(goal_y - y)
            total_delta = np.clip(np.array([delta_x, delta_y, 0., 0.]), -1., 1.)
            obs, _, _, _ = env.step(total_delta)
            x, y, z = obs['observation'][manipulator_idx]
            goal_x, goal_y, goal_z = obs['observation'][object_idx]
            movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
    
        # Open gripper, place it onto object
        open_gripper = np.array([0., 0., 0., 1.])
        env.step(open_gripper)
        while abs(z - goal_z) > 1e-02:
            total_delta = np.clip(np.array([0., 0., 20*(goal_z - z), 0.]), -1., 1.)
            obs, _, _, _ = env.step(total_delta)
            x, y, z = obs['observation'][manipulator_idx]
            goal_x, goal_y, goal_z = obs['observation'][object_idx]
            movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
    
        # Grasp object, raise it up
        for i in range(5):
            env.step(np.array([0., 0., 0., -1]))
            movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))
        
        while z < 0.6:
            obs, _, _, _ = env.step(np.array([0., 0., 1., -1.]))
            x, y, z = obs['observation'][manipulator_idx]
            movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))

        goal_x, goal_y, goal_z = obs['desired_goal'][0]

        # Move object to goal
        while abs(x - goal_x) > 3e-03 or abs(y - goal_y) > 3e-03:
            delta_x, delta_y = 20*(goal_x - x), 20*(goal_y - y)
            total_delta = np.clip(np.array([delta_x, delta_y, 0., -1.]), -1., 1.)
            obs, _, _, _ = env.step(total_delta)
            x, y, z = obs['observation'][manipulator_idx]
            movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))

        # Drop object onto goal
        _, reward, _, _ = env.step(np.array([0., 0., 0., 1.]))
        movie.append(env.render(mode = 'rgb_array', width = video_size, height = video_size))

        print(reward)
    return movie

def save_video(movie, name, video_size = 800):
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (video_size, video_size))
    for frame in movie:
        out.write(frame) 
    out.release()

env = MultipleFetchPickAndPlaceEnv()
env.seed(34)
obs = env.reset()
movie = throw_block(env, obs)
save_video(movie, 'throw.mp4')

env.seed(134)
obs = env.reset()
movie = push_block(env, obs)
save_video(movie, 'push.mp4')

# Interesting seed: 534
env.seed(534)
obs = env.reset()
movie = pick_and_place(env, obs)
save_video(movie, 'pick_and_place.mp4')
print(obs)