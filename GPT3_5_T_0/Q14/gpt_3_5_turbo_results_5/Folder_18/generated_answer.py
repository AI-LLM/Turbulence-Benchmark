
def find_second_smallest_num(numbers):
    sublist = numbers[30:49]
    if len(sublist) < 2:
        return None
    sublist.sort()
    return sublist[1]
