# -*- coding: utf-8 -*-

"""
This is a v2 of the Atari-DQN file which tries to reproduces the google Deepmind Work


This code is based on two main post of two authors:

Ben Lau:

--> https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

Adrien Lucas

--> https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26


"""

__author__ = 'Rubén Mulero'

import gym
import datetime
import os
import json
import random
import numpy as np
import csv
from collections import deque

import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, merge, Input
from keras.layers.merge import Multiply
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten
from keras.optimizers import RMSprop
from keras.utils import plot_model

# Game values
# GAME = 'SpaceInvadersDeterministic-v4'  # Select the game
# GAME = 'DemonAttackDeterministic-v4'
GAME = 'QbertDeterministic-v4'
STATE_SIZE = [105, 80, 4]  # Shape of the images to NN
TARGET_NETWORK_UPDATE = 10000.
# Memory configuration values
REPLAY_MEMORY = 1000000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
# Epsilon values
EXPLORE = 1000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
# Training values
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 50000.  # timesteps to observe before training
MAX_EPISODES = 42000  # Maximun episodes to play the game
MAX_STEPS = 20000000  # Total maximun steps in the experiments

# The program mode TRAIN or simple RUN
MODE = 'train'
# MODE = 'run'

# Keras callbacks to log into TensorBoard
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        embeddings_freq=1
    )
]

# Building model
def atari_model(action_size):
    """
    Buiilds the ATARI MODEL according to the paper of D-QN

    :param action_size: an integer containing the total size of the action space

    :return:
    """
    # Frames input and actions input
    frames_input = Input(STATE_SIZE, name='frames')
    actions_input = Input((action_size,), name='filter')
    # First convolutional layer. here we normalize the input layers between [0, 1]
    conv_1 = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu'
                           )(keras.layers.Lambda(lambda x: x / 255.0)(frames_input))
    # Second convolutional layer
    conv_2 = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(conv_1)
    # Third convolutional layer
    conv_3 = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu')(conv_2)
    # Flatten
    conv_flattened = Flatten()(conv_3)
    # Dense layers
    hidden = Dense(512, activation='relu')(conv_flattened)
    output = Dense(action_size)(hidden)  # Linear
    # Filtering the output by performing a merge
    filtered_output = merge([output, actions_input], mode='mul')
    # TODO to avoid warning check this
    # filtered_output = Multiply()(output, actions_input)
    # Creating the model
    model = Model(input=[frames_input, actions_input], output=filtered_output)
    # TODO test using Adam with LR = 1e-4 with mse loss, but this should be more efficient
    model.compile(optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), loss=huber_loss)
    model.summary()
    return model


def copy_model(model):
    """
    This is a function to store the current model and load it again to make a "copy" of it.

    The idea is to: 1º) Save Model into a file; 2) Clear session; 3) Load again the values in current an target model

    :param model:
    :return:
    """
    # Saving the model
    model.save('tmp_model')
    if K.backend() == 'tensorflow':
        # Deleting model and clearing session
        del model
        K.clear_session()
    # Loading the model again and making a copy
    model = load_model('tmp_model', custom_objects={'huber_loss': huber_loss})
    target_model = load_model('tmp_model', custom_objects={'huber_loss': huber_loss})
    return model, target_model


# Preprocess image
def preprocess_image(img):
    """
    OpenAIGym images are based on 210×160x3.

    We are going to preprocess it to transform this images into 110x84x1 Grayscale

    We can delete the first 26 pixels but convnets can do the hard work

    :param img: a complete 210x160x3 RGB image from OpenAI

    :return: A resized iamge in 110x84x1 Grayscale
    """
    downsampled_image = img[::2, ::2]
    graysacle = np.mean(downsampled_image, axis=2).astype(np.uint8)

    # Test code to see the output
    import matplotlib.pyplot as plt
    # Original image
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # plt.axis("off")
    # plt.savefig('original.png', bbox_inches='tight')
    # Transformed image
    #plt.imshow(graysacle, cmap='gray')
    #plt.savefig('transformed.png', bbox_inches='tight')
    #plt.show()

    return graysacle


# Note: pass in_keras=False to use this function with raw numbers of numpy arrays for testing
def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error * error / 2
    linear_term = abs(error) - 1 / 2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term


def fit_batch(model, target_model, start_states, actions, rewards, next_states, is_terminal):
    """Do one deep Q learning iteration.

    Params:
    - model: The DQN
    - target_model: The target DQN
    - start_states: numpy array of starting states
    - actions: numpy array of one-hot encoded actions corresponding to the start states
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminal: numpy boolean array of whether the resulting state is terminal

    """
    # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    next_Q_values = target_model.predict([next_states, np.ones(actions.shape)])
    # The Q values of the terminal states is 0 by definition, so override them
    next_Q_values[is_terminal] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    Q_values = rewards + GAMMA * np.max(next_Q_values, axis=1)
    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    loss = model.fit(
        [start_states, actions], actions * Q_values[:, None],
        epochs=1, batch_size=len(start_states), verbose=0
    )
    return loss


def run_game(mode='train'):
    # Creating the atari game. Remember to use deterministic values wirh ski frame of k=4
    env = gym.make(GAME)
    # Getting the game action_size
    action_size = env.action_space.n
    # Unwrap the name of the action space
    #env.unwrapped.get_action_meanings()
    # Select what type of action do you want to predict. (useful if some action are unknow or we don't want to predict it)
    discover_actions = np.ones(action_size)
    # Reshape to fit with keras
    discover_actions = np.reshape(discover_actions, [1, action_size])  # Shape 1, n_actions of 1 to predict all.
    # Now we are going to build the model using the action size and the image shape
    model = atari_model(action_size)
    # Plotting the model
    # plot_model(model, show_layer_names=True, show_shapes=True, to_file='dl_model.png')
    # Creating the buffer replay list
    D = deque(maxlen=REPLAY_MEMORY)

    # Creating the CSV file with HEADERS
    if mode == 'train':
        # TODO put this piece of code after ELSE bellow
        with open(r'experiment_results_%s.csv' % GAME, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Date', 'Episode', 'Timestep', "Status", 'Episode Timesteps', 'Episode Reward', 'Max Reward',
                 "Cumulative Reward", "Average Reward", 'Episode Loss', "Weight Updates"])

    #
    # This is only test code to know the mean steps
    #######
    #with open(r'media_pasos_reward_positivo_%s.csv' % GAME, 'a') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(['timestep', 'reward'])
    #last_t = 0
    ######


    # Selecting the mode of the game
    if mode == 'run':
        # Only run the game
        OBSERVE = MAX_EPISODES * OBSERVATION  # Keep a\lways observing. Run mode
        epsilon = FINAL_EPSILON
        print("Now we load weight")
        model.load_weights("model.h5")
        model.compile(optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), loss=huber_loss)
        print("Weight load successfully")
    else:  # We go to training mode
        print("training mode game")
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    # This is the target_model, this should be updated every 10,000 steps
    model, target_model = copy_model(model)
    t = 0  # Timesteps
    max_reward = 0  # Max total reward
    max_steps_reached = False  # Max total steps
    cumulated_reward = 0  # Cumulative reward in the experimentation
    for episode in xrange(0, MAX_EPISODES):
        print("----------Starting episode: ", episode, "----------")
        episode_reward = 0  # Total Episode reward obtained
        episode_steps = 0  # Total Episode steps
        episode_loss = 0  # Total episode loss after some weight updates
        minibatch_update_count = 0  # Total number of weights updates
        # Reset the game and getting the first image
        first_image = env.reset()
        if mode == 'run':
            env.render()
        # Getting the first image stack
        # Preproceesing the first image and stacking in into the first state
        x_t = preprocess_image(first_image)  # 110x84x1
        if episode == 0:
            # The first execution, we create the first 4 image stacks using the same image
            s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # 1*110x84x4
            # Reshape state to fit with Keras needs
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*110*80*4
        else:
            # A new regular episode, adding only the transition of the reset image
            x_t = x_t.reshape(1, x_t.shape[0], x_t.shape[1], 1)  # 1*110x84x1
            s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)  # new 1*110x80x4

        # Playing episode, we will break it if we lose the game
        while True:
            # Selection an action using the e-greedy policy
            a_t = np.zeros([action_size])  # Empty action shape
            if random.random() <= epsilon:
                # print("----------Random Action----------")
                action_index = random.randrange(action_size)
                a_t[action_index] = 1
            else:
                q = model.predict(
                    [s_t, discover_actions])  # input a stack of 4 images, get the prediction and all predictions
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

            # TODO if we implement an Actor-Critic DDPG algoritmh, we need to perform this step at least 4 times
            # Performing the action in the game environment
            new_image, r_t, done, _ = env.step(action_index)
            if mode == 'run':
                env.render()

            # Preprocess the new image and add it into the list
            x_t1 = preprocess_image(new_image)
            # Reshape again to fit with keras
            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x110x84x1
            # Adding to the list and popleft the old one
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)  # new 1x110x84x4
            # Reshape the action to fit with Keras
            a_t = np.reshape(a_t, [1, a_t.shape[0]])
            # store the transition in remember list
            D.append((s_t, a_t, r_t, s_t1, done))
            # Sample minibatch
            if t > OBSERVE:
                # Performing the experience replay
                # Extracting a sample minibach of stored data
                minibatch = random.sample(D, BATCH)
                # Creating empty lists to fit later the model
                state_t = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # numpy array of current state
                action_t = np.zeros((state_t.shape[0], action_size))  # numpy array of performed action
                reward_t = np.zeros(BATCH)  # numpy array of the obtained rewards
                state_t1 = np.zeros((state_t.shape[0], state_t.shape[1], state_t.shape[2],
                                     state_t.shape[3]))  # numpy array of the next state
                terminal = np.full(BATCH, False)  # numpy boolean array of the terminal states

                for i in range(0, len(minibatch)):
                    # state_t = minibatch[i][0]
                    # action_t = minibatch[i][1]      # A one-hot np array with the performed action
                    # reward_t = minibatch[i][2]
                    # state_t1 = minibatch[i][3]
                    # terminal = minibatch[i][4]
                    # Fit the bach model
                    # loss += fit_batch(model, target_model, state_t, action_t, reward_t, state_t1, terminal).history['loss'][0]

                    # storing each transsition
                    state_t[i] = minibatch[i][0]
                    action_t[i] = minibatch[i][1]  # A one-hot np array with the performed action
                    reward_t[i] = minibatch[i][2]
                    state_t1[i] = minibatch[i][3]
                    terminal[i] = minibatch[i][4]

                # Training the model
                loss = fit_batch(model, target_model, state_t, action_t, reward_t, state_t1, terminal).history['loss'][
                    0]
                # Updating the minibach counter per episode
                minibatch_update_count += len(minibatch)
                episode_loss += loss

            # Updating the current state with the new one and adding max reward of the episode
            s_t = s_t1
            t = t + 1
            episode_reward += r_t
            ##
            cumulated_reward += r_t
            episode_steps += 1
            ##

            # We reduced the epsilon gradually
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # Updating the target network every 10000 iterations
            if t % TARGET_NETWORK_UPDATE == 0:
                # Update the target network with the current one
                # print("----------UPDATING TARGET NETWORK----------")
                model, target_model = copy_model(model)

            # save progress every 1000 iterations
            if t % 1000 == 0:
                # print("Now we save model")
                model.save_weights("model.h5", overwrite=True)
                with open("model.json", "w") as outfile:
                    json.dump(model.to_json(), outfile)


            #
            # This is only test code to know the mean steps
            #
            ##########################
            #if t < OBSERVE and r_t > 0:
            #    difference = t - last_t
            #    with open(r'media_pasos_reward_positivo_%s.csv' % GAME, 'a') as f:
            #        writer = csv.writer(f)
            #        writer.writerow([difference, r_t])
            #    last_t = t
            #elif t > OBSERVE:
            #    print("YA HE TERMINADO")
            ##########################


            # Print the info of the current status of the program
            if t <= OBSERVE:
                state = "observe"
            elif OBSERVE < t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            if done or t >= MAX_STEPS:
                # Calculating some needed metrics
                if episode_reward > max_reward:
                    max_reward = episode_reward
                average_reward = episode_reward / episode_steps
                # Save the episode data into a CSV file
                fields = [str(datetime.datetime.now()), episode, t, state, episode_steps, episode_reward, max_reward,
                          cumulated_reward, average_reward, episode_loss, minibatch_update_count]
                with open(r'experiment_results_%s.csv' % GAME, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)

                print("Episode finished!")
                print("************************")
                print ("EPISODE STEPS", episode_steps, "EPISODE REWARD", episode_reward, "/MAX_REWARD ", max_reward,
                       "/CUMULATIVE REWARD", cumulated_reward, "/LOSS", episode_loss)
                print("************************")

                # If the reason of being here is because we reached the maximun steps, we are going to force finish.
                if t >= MAX_STEPS:
                    # We have reached the maximum step size of this experiment
                    print("Reached maximum steps. Finishing the experiment.")
                    max_steps_reached = True

                break

        # Forcing exiting of the main loop
        if max_steps_reached:
            break

    # End situation and saving results in a CSV file
    # Saving the last iteration of the model
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

    print("************************")
    print("Test finished successfully")
    print("************************")
    print("*********END************")


if __name__ == "__main__":
    # Use only 1 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Loading tensroflow configuration
    config = tf.ConfigProto()
    # TensorFlow optimizations
    config.gpu_options.allow_growth = True  # Allow to use only the needed GPU memory
    sess = tf.Session(config=config)
    # Setting the keras session
    K.set_session(sess)
    # Creating the log directory if it not exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    # Standard Python RUN
    run_game(mode=MODE)
