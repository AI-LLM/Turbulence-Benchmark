
def identical_elements(list1, list2):
    # Create a set of all elements in both lists
    common_elements = set(list1).intersection(set(list2))
    # Initialize an empty set to store the results
    result = set()
    # Iterate over the common elements and check if they occur at least once in both lists between index 70 and index 200, both inclusive
    for element in common_elements:
        count1 = list1.count(element)
        count2 = list2.count(element)
        # If the element occurs at least once in both lists between index 70 and index 200, both inclusive, add it to the result set
        if count1 > 0 and count2 > 0:
            for i in range(70, 201):
                if list1[i] == element or list2[i] == element:
                    result.add(element)
    # Return the set of all elements that occur at least once in both lists between index 70 and index 200, both inclusive
    return result
