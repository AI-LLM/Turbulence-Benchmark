
def all_neg_ints_inclusive(my_list):
    neg_ints = []
    for i in range(74, 96):
        if my_list[i] < 0:
            neg_ints.append(my_list[i])
    return neg_ints
