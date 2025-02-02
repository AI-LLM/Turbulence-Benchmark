
def palindromes_of_specific_lengths(string):
    # Find all palindromes in the substring from index 75 to index 95, both inclusive
    palindromes = []
    for i in range(75, 96):
        for j in range(i + 1, min(len(string), 96)):
            if string[i:j] == string[i:j][::-1]:
                palindromes.append(string[i:j])
    # Filter out palindromes that are not between lengths 7 and 9 (both inclusive)
    filtered_palindromes = []
    for palindrome in palindromes:
        if len(palindrome) >= 7 and len(palindrome) <= 9:
            filtered_palindromes.append(palindrome)
    # Filter out palindromes that do not only consist of English letters
    final_palindromes = []
    for palindrome in filtered_palindromes:
        if all(letter in "abcdefghijklmnopqrstuvwxyz" for letter in palindrome):
            final_palindromes.append(palindrome)
    return set(final_palindromes)
