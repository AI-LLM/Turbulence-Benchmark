
def identical_elements(list1, list2):
    # Initialize an empty set to store the identical elements
    identical_set = set()
    # Loop through both lists simultaneously
    for i in range(len(list1)):
        # Check if the current element is also present in the other list
        if list1[i] in list2:
            # If it is, add it to the set of identical elements
            identical_set.add(list1[i])
    return identical_set
