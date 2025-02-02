
def identical_elements(list1, list2):
    # Create a set of all elements in both lists
    common_elements = set(list1).intersection(set(list2))
    # Filter out elements that do not occur at least once in both lists between index 70 and index 84 (inclusive)
    filtered_elements = [element for element in common_elements if element in list1[70:85] or element in list2[70:85]]
    return set(filtered_elements)
