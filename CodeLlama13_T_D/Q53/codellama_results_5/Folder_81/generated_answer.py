
def palindrome_of_length_at_least_n(string):
    # Find all possible palindromes in the string
    palindromes = set()
    for i in range(len(string)):
        for j in range(i, len(string)):
            if string[i] == string[j]:
                palindromes.add(string[i:j+1])
    
    # Filter out palindromes that are not at least 33 characters long
    filtered_palindromes = set()
    for palindrome in palindromes:
        if len(palindrome) >= 33:
            filtered_palindromes.add(palindrome)
    
    return filtered_palindromes
