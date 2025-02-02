
def identical_elements(list1, list2):
    # create a set of all elements in both lists
    all_elements = set(list1).union(set(list2))
    # initialize an empty set to store the result
    identical_elements = set()

    # loop through each element in the set of all elements

    for element in all_elements:
        # check if the element occurs at least once in both lists

        if element in list1 and element in list2:
            # add the element to the set of identical elements

            identical_elements.add(element)

    # return the set of identical elements

    return identical_elements
