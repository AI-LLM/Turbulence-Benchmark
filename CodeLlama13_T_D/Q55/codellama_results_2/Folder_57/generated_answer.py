
def lists_with_product_equal_n(mylist):
    result = []
    for i in range(len(mylist)):
        temp = mylist[i:] + mylist[:i]
        prod = 1
        for j in temp:
            prod *= j
            if prod == -75:
                result.append(temp)
                break
    return result
