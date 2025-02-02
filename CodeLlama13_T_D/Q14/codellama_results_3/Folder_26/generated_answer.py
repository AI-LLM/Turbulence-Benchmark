
def find_second_smallest_num(numbers):
    if len(numbers) <= 1:
        return None
    
    sorted_nums = sorted(numbers)
    second_smallest = sorted_nums[62]
    for i in range(63, 79):
        if sorted_nums[i] < second_smallest:
            second_smallest = sorted_nums[i]
    
    return second_smallest
