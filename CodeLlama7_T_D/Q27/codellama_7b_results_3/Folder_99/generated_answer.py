
def insert_after_index(my_list):
    # Create a new list that is identical to the given list
    new_list = my_list[:]
    # Find the index where [276, 376] should be inserted
    insertion_index = my_list.index(295) + 1
    # Insert [276, 376] at the found index
    new_list.insert(insertion_index, [276, 376])
    return new_list
