def all_even_ints_exclusive(nums):
    return [x for x in range(361, 643) if x % 2 == 0 and x in nums]
