
def find_second_smallest_num(numbers):
    if len(numbers) < 2:
        return None
    second_smallest = numbers[0]
    for i in range(1, len(numbers)):
        if numbers[i] < second_smallest:
            second_smallest = numbers[i]
    return second_smallest
