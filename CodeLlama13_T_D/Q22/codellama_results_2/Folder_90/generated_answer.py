
def find_sum_of_multiples(integer):
    return sum([i for i in range(1, 318*integer+1) if i % integer == 0])
