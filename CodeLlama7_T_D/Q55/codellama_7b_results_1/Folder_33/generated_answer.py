
def lists_with_product_equal_n(circular_list):
    # Initialize an empty list to store the sublists
    sublists = []

    # Iterate over the elements of the circular list
    for i in range(len(circular_list)):
        # Initialize a variable to store the product of the current sublist
        product = 1

        # Iterate over the elements of the current sublist
        for j in range(i, len(circular_list)):
            # Multiply the current element by the product of the previous elements
            product *= circular_list[j]

            # If the product is equal to -115, add the sublist to the list of sublists
            if product == -115:
                sublists.append(circular_list[i:j+1])

    # Return the list of sublists that have a product equal to -115
    return sublists
