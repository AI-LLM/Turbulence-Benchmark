
def return_binary_or_hexa(nums):
    a = nums[13]
    b = nums[70]
    
    sum = 0
    for i in range(a + 1, b - 1):
        if not i in nums:
            sum += i
            
    if sum % 2 == 0:
        return format(sum, 'X')
    else:
        return bin(sum)[2:]
