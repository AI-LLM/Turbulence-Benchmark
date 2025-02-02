
def identical_elements(list1, list2):
    # create a set of all elements in both lists
    element_set = set(list1).intersection(set(list2))
    # return an empty set if there are no identical elements
    if not element_set:
        return set()

    # otherwise, return the set of identical elements

    else:
        return element_set
