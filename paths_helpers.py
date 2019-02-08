import pathlib
from functools import reduce
from operator import truediv
from itertools import chain


def get_configuration_dir(c):
    return _get_path(
        f"ill_{int(c.allow_illegal)}", f"em{c.min_eps:.2f}", c.optimizer, c.loss, c.conv_activation, c.layers_size,
        c.output_activation, f"batch_{c.batch_size}", f"t_upd_freq{c.update_targets_each}",
        f"learn_freq_{c.learn_each}", f"mem_{c.memory_size}", c.state_map_function, f"ddq_{int(c.double_q)}",
        c.reward_func, f"eps_c_{c.epsilon_constant}", f"eq_dones{int(c.equal_dones)}",
        f"eq_dirs{int(c.equal_directions)}", f"cr{int(c.crucial)}", f"dry{int(c.dry)}"
    )


def get_configuration_one_dir(c):
    mem = ('S' if c.equal_directions else '') + 'RM' + ('D' if c.equal_dones else '') + (
        'C' if c.crucial else '') + str(c.memory_size)

    params = (
        f"ill{int(c.allow_illegal)}", f"em{c.min_eps:.2f}".replace('.', ''), c.optimizer, c.loss, c.conv_activation,
        _format_conv_layers_info(c.conv_layers), c.layers_size, c.output_activation, f"batch_{c.batch_size}",
        f"tUpdF{c.update_targets_each}", f"learnF{c.learn_each}", c.state_map_function, f"ddq_{int(c.double_q)}",
        c.reward_func, f"epsC{c.epsilon_constant}", mem, f"dry{int(c.dry)}")

    return '_'.join(map(format_for_path, params))


def _get_path(*args):
    return reduce(truediv, map(format_for_path, args), pathlib.Path())


def _format_conv_layers_info(conv_params):
    return '_'.join(map(str, chain.from_iterable(conv_params)))


def format_for_path(param):
    if hasattr(param, 'short_name') and param.short_name is not None:
        return param.short_name
    if callable(param):
        return param.__name__
    if isinstance(param, list):
        return '_'.join(str(val) for val in param)
    return str(param)


def get_max_number_file(path: pathlib.Path):
    return max([int(p.with_suffix('').name) for p in path.iterdir() if p.with_suffix('').name.isdigit()] + [0])
