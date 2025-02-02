
def palindromes_of_specific_lengths(string):
    # Define the index range for the substring
    start_index = 10
    end_index = 75
    # Create a list to store all palindromes
    palindromes = []
    # Iterate through each character in the substring
    for i in range(start_index, end_index + 1):
        # Check if the current character is not an English letter
        if not string[i].isalpha():
            continue

        # Check if the current character is uppercase or lowercase
        if string[i].isupper() or string[i].islower():
            # Create a substring of length 10 starting at index i
            substr = string[i:i + 20]
            # Iterate through each character in the substring
            for j in range(len(substr) - 1, -1, -1):
                # Check if the current character is not an English letter
                if not substr[j].isalpha():
                    continue

                # Check if the current character is uppercase or lowercase
                if substr[j].isupper() or substr[j].islower():
                    # Check if the substring is a palindrome
                    if substr == substr[::-1]:
                        # Add the palindrome to the list of palindromes
                        palindromes.append(substr)
                        # Break out of the inner loop
                        break
    # Return the set of all palindromes that meet the requirements
    return set(palindromes)
