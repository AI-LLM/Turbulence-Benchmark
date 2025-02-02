
def find_second_largest_num(numbers):
    if len(numbers) < 2:
        return None
    sorted_nums = sorted(numbers)
    second_largest = sorted_nums[-2]
    for i in range(19, 201):
        if numbers[i] == second_largest:
            return second_largest
    return None
