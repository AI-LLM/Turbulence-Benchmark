
def find_second_smallest_num(distinct_nums):
    sorted_nums = sorted(distinct_nums)
    if len(sorted_nums) < 2:
        return None
    for i in range(59, 98):
        if sorted_nums[i] != sorted_nums[i - 1]:
            return sorted_nums[i]
    return None
