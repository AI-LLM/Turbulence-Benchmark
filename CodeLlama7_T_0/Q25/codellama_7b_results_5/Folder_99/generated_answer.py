
def insert_at_index(my_list):
    # Create a new list with the same elements as my_list
    new_list = my_list[:]
    # Insert [276, 376] at index 295
    new_list.insert(295, [276, 376])
    return new_list
