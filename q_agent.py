import random
import itertools

from neural_network import prepare_training_batch, format_for_input, one_hot_encode_input, is_model_one_hot
from configurations import *

Experience = namedtuple('Experience', 'from_state action reward to_state done')

# TODO: try one-hot board
# TODO: try convolutions
# TODO: try action as an input neuron
# TODO: try old batch approach
# TODO: what to change in the network
# TODO: optimizer params
# TODO: huber loss params
# TODO: how to debug network, will tensor-board tell me anything in this case

DIAG_MSG_FREQ = 1000


class DQNAgent:
    def __init__(self, start_step, model, target_model, epsilon_func, memory_keeper, batch_size, learn_each,
                 discount_factor, update_target_each, strategy, out_dir):
        self.model = model
        self.target_model = target_model
        self.step_counter = start_step
        self.epsilon_func = epsilon_func
        self.epsilon = epsilon_func(self.step_counter)
        self.memory = memory_keeper
        self.batch_size = batch_size
        self.learn_each_n_steps = learn_each
        self.discount_factor = discount_factor
        self.update_target_each = update_target_each
        self.diagnostic_file_counter = itertools.count()
        self.train_counter = itertools.count()
        self.train_number = 0
        self._dqn = target_model is not model
        self.strategy = strategy
        self.out_dir = out_dir
        self._input_format_func = one_hot_encode_input if is_model_one_hot(self.model) else format_for_input

    def act(self, state, choices):
        if random.random() < self.epsilon:
            return random.choice(choices)
        return choices[0] if len(choices) == 1 else self.strategy(self.get_q_values(state), choices)

    def get_q_values(self, state):
        return self.model.predict(self._input_format_func([state]), batch_size=1)[0]

    def one_full_step(self, choices, c, env, forget, dry=False):
        mapped_from_state = c.state_map_function(env.state())
        move = self.act(mapped_from_state, choices)
        if dry:
            step_result = env.dry_step(move)
        else:
            step_result = env.step(move)
        reward = c.reward_func(step_result)
        if not forget:
            self.remember(
                Experience(mapped_from_state, move, reward, c.state_map_function(step_result.state),
                           step_result.is_game_over))
            history = self.finish_step()
        return step_result, reward, move

    def play_until_made_move(self, c, env, forget):
        made_move = False
        choices = [0, 1, 2, 3]
        while not made_move:
            step_result, reward, move = self.one_full_step(choices, c, env, forget)
            choices.remove(move)
            made_move = step_result.board_move_result.move_count != 0
        return step_result, reward

    def test_all_do_best(self, c, env, forget):
        mapped_from_state = c.state_map_function(env.state())
        real_move_to_make = self.act(mapped_from_state, env.act_space())
        dry_choices = (i for i in range(4) if i != real_move_to_make)
        for dry_move in dry_choices:
            self.one_full_step([dry_move], c, env, forget, dry=True)
        return self.one_full_step([real_move_to_make], c, env, forget)

    def remember(self, experience):
        self.memory.append(experience)

    def finish_step(self):
        self.step_counter += 1
        self.epsilon = self.epsilon_func(self.step_counter)
        if self.step_counter % self.learn_each_n_steps == 0 and len(self.memory) >= self.batch_size:
            return self._train()

    def percentage_memory_stats(self):
        return self.memory.percentage_memory_stats()

    def _train(self):
        self.train_number = next(self.train_counter)
        random_sample = self.memory.sample(self.batch_size)
        if len(random_sample) < self.batch_size:
            return
        x_train, y_train = prepare_training_batch(random_sample, self.model,
                                                  self.target_model, self.discount_factor, self._input_format_func)
        if self._dqn and self.train_number % self.update_target_each == 0:
            self.target_model.set_weights(self.model.get_weights())

        return self.model.fit(x_train, y_train, batch_size=len(x_train), epochs=1, shuffle=True, verbose=False)
