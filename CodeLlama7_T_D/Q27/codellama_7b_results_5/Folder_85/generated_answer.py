
def insert_after_index(my_list):
    # Create a new list with the same elements as my_list
    new_list = my_list[:]
    # Insert 304.62 at the index immediately succeeding index 51
    new_list.insert(new_list.index(51) + 1, 304.62)
    return new_list
