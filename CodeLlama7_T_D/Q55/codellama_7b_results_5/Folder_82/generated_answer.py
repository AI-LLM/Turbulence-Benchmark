
def lists_with_product_equal_n(circular_list):
    # Create a new empty list to store the sublists
    sublists = []

    # Iterate through the circular list and find contiguous integers whose product equals 8

    for i in range(len(circular_list)):
        current_product = 1
        start_index = i

        while current_product == 1 and i < len(circular_list):
            current_product *= circular_list[i]
            i += 1

        if current_product == 8:
            sublists.append(circular_list[start_index:i]))

    return sublists
