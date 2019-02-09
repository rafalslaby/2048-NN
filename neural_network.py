from typing import List
import os
import numpy as np
from keras.layers import Dense, Flatten, InputLayer, Conv2D
from keras.models import Sequential
from configurations import *


def make_model(deep_layer_sizes: List[int], optimizer='Adam', output_activation='softmax', loss='mse',
               conv_layers=[], conv_activation=None):
    model = Sequential()
    model.add(InputLayer(input_shape=(4, 4, 1)))
    for filters, kernel_size in conv_layers:
        model.add(Conv2D(filters, kernel_size, activation=conv_activation, ))
    model.add(Flatten())
    for size in deep_layer_sizes:
        model.add(Dense(size, activation='relu'))
    model.add(Dense(4, activation=output_activation))
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['mse'])
    return model


def format_for_input(arr):
    return np.array(arr).reshape(-1, 4, 4, 1)


def prepare_training_batch(experiences, model, target_model, gamma):
    to_states = []
    from_states = []
    actions = []
    rewards = []
    dones = []
    for exp in experiences:
        to_states.append(exp.to_state)
        from_states.append(exp.from_state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.done)
    next_q_tables = target_model.predict(format_for_input(to_states), batch_size=len(experiences))
    target_q_tables = model.predict(format_for_input(from_states), batch_size=len(experiences))
    rewards = np.array(rewards)
    actions = np.array(actions)
    dones = np.array(dones)

    target_q_tables[np.arange(len(experiences)), actions] += LR * (
            rewards + (1 - dones) * gamma * np.max(next_q_tables, axis=1) -
            target_q_tables[np.arange(len(experiences)), actions])

    return format_for_input(from_states), np.array(target_q_tables)


def prepare_training_batch_iter(experiences, model, target_model, gamma):
    """
    Slower version ;/
    """
    to_states = []
    from_states = []
    for exp in experiences:
        to_states.append(exp.to_state)
        from_states.append(exp.from_state)

    next_q_tables = target_model.predict(np.array(to_states), batch_size=len(experiences))
    target_q_tables = model.predict(np.array(from_states), batch_size=len(experiences))
    for i in range(len(experiences)):
        target_value = experiences[i].reward + (1 - experiences[i].done) * gamma * np.max(next_q_tables[i])
        target_q_tables[i][experiences[i].action] = target_value

    return np.array(from_states), np.array(target_q_tables)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = make_model([8] * 6)
    n = 10000
    from time import time

    single_x = np.random.random((1, 4, 4))
    single_y = np.random.random((1, 4))
    multiple_x = np.random.random((n, 4, 4))
    multiple_y = np.random.random((n, 4))
    s = time()
    for i in range(n):
        model.predict(single_x, batch_size=1)
    print(time() - s)
    s = time()
    model.predict(multiple_x, batch_size=n)
    print(time() - s)
    s = time()
    for i in range(n):
        model.fit(single_x, single_y, batch_size=1, verbose=False)
    print(time() - s)
    s = time()
    model.fit(multiple_x, multiple_y, batch_size=n, verbose=False)
    print(time() - s)
