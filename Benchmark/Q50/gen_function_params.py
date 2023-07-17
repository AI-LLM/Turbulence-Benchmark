import random
import string
import itertools


# This function generates a list of random strings. The output of this function will be passed to the generated code
# by the large language model and the model solution.
def input_generator(l, seed):
    if seed != 'default':
        random.seed(seed)
    result = []
    n = int(l[0])
    option = random.randint(0, 3)
    if option == 0:
        for _ in range(2 * (n + 1)):
            length = random.randint(3, 5)
            result.append(''.join(random.choices(string.ascii_letters, k=length)))
    else:
        if n > 276:
            s = ''.join(random.choices(string.ascii_letters, k=5))
            result = [''.join(i) for i in itertools.permutations(s)]
        elif 16 < n <= 276:
            s = ''.join(random.choices(string.ascii_letters, k=4))
            result = [''.join(i) for i in itertools.permutations(s)]
        else:
            s = ''.join(random.choices(string.ascii_letters, k=3))
            result = [''.join(i) for i in itertools.permutations(s)]
        for _ in range(n // 5):
            length = random.randint(3, 5)
            result.append(''.join(random.choices(string.ascii_letters, k=length)))

    random.shuffle(result)
    return result