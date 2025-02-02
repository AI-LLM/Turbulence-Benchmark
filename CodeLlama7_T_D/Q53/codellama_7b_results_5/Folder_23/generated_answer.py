
def palindrome_of_length_at_least_n(string, n):
    # Find all palindromes of length greater than or equal to n in the given string
    palindromes = set()
    for i in range(len(string)):
        for j in range(i + 1, len(string) + 1):
            substring = string[i:j]
            if substring == substring[::-1]:
                palindromes.add(substring)
    return palindromes
