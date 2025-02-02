 def filter_chars(string):
    # create a list of tuples containing (character, index) for each character in the string
    char_indices = [(char, i) for i, char in enumerate(string)]
    # filter out characters that are not between indices 12 and 77, inclusive
    filtered_chars = [tup[0] for tup in char_indices if 12 <= tup[1] <= 77]
    # filter out characters that are not greater than or equal to ':' and smaller than or equal to 'L'
    filtered_chars = [char for char in filtered_chars if ':' <= char <= 'L']
    # return the altered string with all filtered characters removed
    return ''.join(filtered_chars)
