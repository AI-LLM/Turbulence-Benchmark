
def identical_elements(list1, list2):
    # Create a set of all elements in both lists between indices 75 and 85, inclusive
    element_set = set(list1[75:86]) & set(list2[75:86])
    return element_set
