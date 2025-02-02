
def lists_with_product_equal_n(circular_list):
    # Initialize an empty list to store the sublists
    sublists = []

    # Iterate over the circular list and check if the product of a contiguous sequence of integers is equal to 3
    for i in range(len(circular_list)):
        # Calculate the product of a contiguous sequence of integers starting from index i
        prod = 1
        for j in range(i, len(circular_list)):
            prod *= circular_list[j]

            # If the product is equal to 3, add the sublist to the list of sublists
            if prod == 3:
                sublists.append(circular_list[i:j+1])
                # Break out of the inner loop and move on to the next index in the outer loop
                break

    # Return the list of sublists

    return sublists
