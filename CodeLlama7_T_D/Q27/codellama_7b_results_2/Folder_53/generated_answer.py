
def insert_after_index(my_list):
    # create a new list with the same elements as my_list
    new_list = []
    # iterate over the elements of my_list
    for i in range(len(my_list)):
        # if the current index is 92, insert 709.87 at the next index
        if i == 92:
            new_list.append(my_list[i])
            new_list.append(709.87)
        else:
            new_list.append(my_list[i])
    # return the new list with the inserted element
    return new_list
