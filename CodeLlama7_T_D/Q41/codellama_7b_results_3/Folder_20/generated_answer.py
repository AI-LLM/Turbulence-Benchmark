
def identical_elements(list1, list2):
    # create a set of all elements in both lists between indices 56 and 66, inclusive
    common_elements = set(list1[56:67]).intersection(set(list2[56:67]))
    return common_elements
