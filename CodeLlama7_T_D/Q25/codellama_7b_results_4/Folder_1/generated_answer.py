
def insert_at_index(my_list):
    # Create a new list with an extra element
    new_list = my_list[:93] + [33, 78] + my_list[93:]
    return new_list
