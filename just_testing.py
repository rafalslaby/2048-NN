import numpy as np


def choose_best_valid(q_table, choices):
    i_max = choices[0]
    for i in range(len(q_table)):
        if q_table[i] > q_table[i_max] and i in choices:
            i_max = i
    return i_max


def get_random_choices():
    return np.random.choice([[0, 1, 2, 3], [0, 1], [0]], p=[0.9, 0.05, .05])


import cProfile


# cProfile.run('for i in range(100000): choose_best_valid(np.random.random(4),get_random_choices())')


def manual_histogram(x):
    histogram = [0] * 4
    for i in x:
        histogram[i] += 1
    return histogram


def numpy_histogram(x):
    return np.histogram(x, [-0.5, 0.5, 1.5, 2.5, 3.5])


def np_unique(x):
    return np.unique(x, return_counts=True)


# x = list(np.random.random_integers(0, 3, 1_000_000_0))
# print("Started")
# start = time.time()
# np.array(x)
# print(time.time() - start)
#
# start = time.time()
# print(np_unique(x))
# print(time.time() - start)
from collections import namedtuple

Experience = namedtuple('Experience', 'from_state action reward to_state done')

experiences = [Experience(np.random.random((4, 4)), 1, 100, np.random.random((4, 4)), 0) for i in range(2)]
x = np.array(experiences)
print(np.mean(np.mean(x[:,0])))
# print(np.mean(x[:,1]))


# def set_test(n):
#     s = set()
#     for i in range(n):
#         e = Experience(np.zeros((4, 4), dtype='int8'), i, i, np.zeros((4, 4), dtype='int8'))
#         s.add(e)
#     print(getsizeof(s) * 1e-6)
#
#
# def list_test(n):
#     s = []
#     for i in range(n):
#         e = Experience(np.zeros((4, 4), dtype='int8'), i, i, np.zeros((4, 4), dtype='int8'))
#         s.append(e)
#     print(getsizeof(s) * 1e-6)

# import timeit
# print(timeit.timeit('set_test(100000)',number=1, globals=globals()))
# print(timeit.timeit('list_test(100000)',number=1, globals=globals()))
#
# import cProfile
#
# import pstats
# cProfile.run('set_test(100000)','stats')
# p = pstats.Stats('stats').sort_stats('cumulative')
# p.print_stats(20)
#
# cProfile.run('list_test(100000)','stats')
# p = pstats.Stats('stats').sort_stats('cumulative')
# p.print_stats(20)