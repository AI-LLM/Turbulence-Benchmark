
def gcf_two_nums(nums):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    num1 = nums[28]
    num2 = nums[17]
    
    return gcd(num1, num2)
