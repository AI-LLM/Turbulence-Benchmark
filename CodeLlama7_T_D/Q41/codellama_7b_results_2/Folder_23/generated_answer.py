
def identical_elements(list1, list2):
    # Initialize a set to store the identical elements
    identical_set = set()
    # Iterate over both lists simultaneously
    for i in range(len(list1)):
        # Check if the current element is in both lists
        if list1[i] in list2 and list2[i] in list1:
            # Add the identical element to the set
            identical_set.add(list1[i])
    return identical_set
