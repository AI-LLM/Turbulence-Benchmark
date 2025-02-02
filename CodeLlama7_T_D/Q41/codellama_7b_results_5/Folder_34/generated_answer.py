
def identical_elements(list1, list2):
    # Create a set of all elements in both lists
    all_elements = set(list1).union(set(list2))
    # Initialize an empty set to store the result
    identical_elements = set()

    # Iterate over each element in the union of both lists
    for element in all_elements:
        # Check if the element occurs at least once in both lists between index 60 and index 200, inclusive
        if (element in list1[60:200]) and (element in list2[60:200])):
            identical_elements.add(element)
    # Return the set of all identical elements found between index 60 and index 200, inclusive
    return identical_elements
