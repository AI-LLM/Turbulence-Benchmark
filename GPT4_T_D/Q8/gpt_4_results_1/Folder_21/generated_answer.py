
def all_even_ints_exclusive(int_list):
    even_ints = []
    for index in range(744, 867):
        if index < len(int_list) and int_list[index] % 2 == 0:
            even_ints.append(int_list[index])
    return even_ints
