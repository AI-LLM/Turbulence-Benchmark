def return_binary_or_hexa(test_tup):
    res = test_tup[1] - test_tup[0] - 1
    if res % 2 == 0:
        return hex(res)
    else:
        return bin(res)
