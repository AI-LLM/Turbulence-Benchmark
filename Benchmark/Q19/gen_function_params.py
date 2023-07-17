import random


# This function generates a list of random integers. The output of this function will be passed to the generated code
# by the large language model and the model solution.
def input_generator(l, seed):
    if seed != 'default':
        random.seed(seed)
    x = int(l[-1]) + 2
    random_selection = random.randint(0, 1)
    if random_selection:
        result = random.choices(range(-550, 550), k=x)
    else:
        result = random.sample(range(-550, 550), k=x)

    random.shuffle(result)
    return result
