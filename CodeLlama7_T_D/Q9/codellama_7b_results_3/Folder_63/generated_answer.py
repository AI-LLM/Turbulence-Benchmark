
def all_odd_ints_inclusive(my_list):
    odd_ints = []
    for i in range(22, 89):
        if my_list[i] % 2 != 0:
            odd_ints.append(my_list[i])
    return odd_ints
