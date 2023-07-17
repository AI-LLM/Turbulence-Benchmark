import random


# This function generates a list of random integers. The output of this function will be passed to the generated code
# by the large language model and the model solution.
def input_generator(l, seed):
    if seed != 'default':
        random.seed(seed)
    x = int(l[1]) + 2
    result = []
    for _ in range(2):
        random_selection = random.randint(0, 1)
        if random_selection:
            result += random.choices(range(-1000, 1000), k=x)
        else:
            result += random.sample(range(-1000, 1000), x)

    random.shuffle(result)
    return result
