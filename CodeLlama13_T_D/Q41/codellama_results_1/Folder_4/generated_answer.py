
def identical_elements(list1, list2):
    elements = set()
    for i in range(12, 93):
        if list1[i] == list2[i]:
            elements.add(list1[i])
    return elements
