
def lists_with_product_equal_n(circular_list):
    # Initialize an empty list to store the sublists
    sublists = []

    # Iterate through the circular list and find contiguous integers with product equal to -15
    for i in range(len(circular_list)):
        product = 1

        # Calculate the product of the current element and its predecessor
        if i > 0:
            product *= circular_list[i - 1]

        # Calculate the product of the current element and its successor
        if i < len(circular_list) - 1:
            product *= circular_list[i + 1]

        # If the product is equal to -15, add the sublist to the list of sublists
        if product == -15:
            sublists.append(circular_list[i:i + 2])

    # Return the list of sublists
    return sublists
