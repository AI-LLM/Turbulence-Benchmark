
def palindrome_of_length_at_least_n(string):
    # Initialize an empty set to store the palindromes
    palindromes = set()
    
    # Iterate over each character in the string
    for i in range(len(string)):
        # Check if the substring starting from the current index is a palindrome
        if is_palindrome(string[i:]):
            # Add the palindrome to the set
            palindromes.add(string[i:])
    
    return palindromes

def is_palindrome(string):
    # Check if the string is a palindrome by comparing it with its reverse
    return string == string[::-1]
