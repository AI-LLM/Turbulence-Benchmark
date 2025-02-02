
def palindrome_of_length_at_least_n(string):
    # Create a set to store the palindromes
    palindromes = set()
    
    # Iterate over each substring of length at least 45 in the string
    for i in range(len(string) - 44):
        substring = string[i:i+45]
        
        # Check if the substring is a palindrome
        if substring == substring[::-1]:
            palindromes.add(substring)
            
    return palindromes
