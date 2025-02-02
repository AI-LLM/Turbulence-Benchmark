
def identical_elements(list1, list2):
    # Initialize a set to store the identical elements
    identical_elements = set()

    # Loop through both lists and check if any element occurs in both lists
    for i in range(len(list1)):
        for j in range(len(list2)):
            if list1[i] == list2[j]:
                identical_elements.add(list1[i])

    # Return the set of all identical elements
    return identical_elements
