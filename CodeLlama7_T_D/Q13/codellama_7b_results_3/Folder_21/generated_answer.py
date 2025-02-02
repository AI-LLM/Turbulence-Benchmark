
def find_second_largest_num(my_list):
        # Check if list is empty
        if len(my_list) == 0:
            return None

        # Sort the list in ascending order
        my_list.sort()

        # Initialize variables to keep track of the second largest element and its index
        second_largest = None
        second_largest_index = -1

        # Iterate through the list and find the second largest element
        for i in range(len(my_list)):
            if my_list[i] > second_largest:
                second_largest = my_list[i]
                second_largest_index = i

        # Return the second largest element from index 639 to index 975, both inclusive
        return my_list[second_largest_index - 638:second_largest_index + 1 - 974]
