
def sum_ints_div_by_either_nums(arr):
    result = 0
    for i in range(25, 81):
        if arr[i] % -20 == 0 or arr[i] % -26 == 0:
            result += arr[i]
    return result
