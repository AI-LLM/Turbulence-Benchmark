def sum_in_range(nums):
    return sum(nums[i] for i in range(len(nums)) if abs(nums[i]) <= 93)
