
def lists_with_product_equal_n(lst):
    result = []
    for i in range(len(lst)):
        prod = 1
        temp = []
        for j in range(i, len(lst)+i):
            prod *= lst[j%len(lst)]
            temp.append(lst[j%len(lst)])
            if prod == 53:
                result.append(temp)
                break
    return result
