# A python function that returns the list of all integers from x to y both inclusive.
def all_ints_inclusive(x,y):
    return [i for i in range(x, y+1) if x <= y]


# A python function that returns the list of all integers from x to y both exclusive.
def all_ints_exclusive(x,y):
    return [i for i in range(x+1, y) if x < y]


# A python function that returns the list of all positive integers from x to y both inclusive.
def all_pos_ints_inclusive(x,y):
    return [i for i in range(x, y+1) if x <= y and i > 0]


# A python function that returns the list of all positive integers from x to y both exclusive.
def all_pos_ints_exclusive(x,y):
    return [i for i in range(x+1, y) if x < y and i > 0]


# A python function that returns the list of all negative integers from x to y both inclusive.
def all_neg_ints_inclusive(x,y):
    return [i for i in range(x, y+1) if x <= y and i < 0]


# A python function that returns the list of all negative integers from x to y both exclusive.
def all_neg_ints_exclusive(x,y):
    return [i for i in range(x+1, y) if x < y and i < 0]


# A python function that returns the list of all even integers from x to y both inclusive.
def all_even_ints_inclusive(x,y):
    return [i for i in range(x, y+1) if x <= y and i%2==0]


# A python function that returns the list of all even integers from x to y both exclusive.
def all_even_ints_exclusive(x,y):
    return [i for i in range(x+1, y) if x < y and i%2==0]


# A python function that returns the list of all odd integers from x to y both inclusive.
def all_odd_ints_inclusive(x,y):
    return [i for i in range(x, y+1) if x <= y and i%2!=0]


# A python function that returns the list of all odd integers from x to y both exclusive.
def all_odd_ints_exclusive(x,y):
    return [i for i in range(x+1, y) if x < y and i%2!=0]


# A python function that returns the list of all integers from x to y that are divisible by n. Both x and y should be inclusive.
def all_ints_div_by_n_inclusive(x,y,n):
    return [i for i in range(x, y+1) if x <= y and i%n==0]


# A python function that returns the list of all integers from x to y that are divisible by n. Both x and y should be exclusive.
def all_ints_div_by_n_exclusive(x,y,n):
    return [i for i in range(x+1, y) if x < y and i%n==0]


# A python function that returns the list of all integers from x to y that are not divisible by n. Both x and y should be inclusive.
def all_ints_not_div_by_n_inclusive(x,y,n):
    return [i for i in range(x, y+1) if x <= y and i%n!=0]


# A python function that returns the list of all integers from x to y that are divisible by m and n. Both x and y should be inclusive.
def all_ints_div_by_m_n_inclusive(x,y,m,n):
    return [i for i in range(x, y+1) if x <= y and i%m==0 and i%n==0]


# A python function that returns the list of all integers from x to y that are divisible by m or by n. Both x and y should be inclusive.
def all_ints_div_by_m_n_inclusive(x,y,m,n):
    return [i for i in range(x, y+1) if x <= y and i%m==0 or i%n==0]


# A python function that returns the sum of all integers from x to y inclusive.
def sum_ints(x,y):
    return sum(range(x,y+1))


# A python function that returns the sum of all even integers from x to y inclusive.
def sum_even_ints(x,y):
    return sum([i for i in range(x,y+1) if i%2==0])


# A python function that returns the sum of all odd integers from x to y inclusive.
def sum_odd_ints(x,y):
    return sum([i for i in range(x,y+1) if i%2!=0])


# A python function that returns the sum of all integers from x to y that are divisible by n. Both x and y are inclusive.
def sum_ints_div_by_n(x,y,n):
    return sum([i for i in range(x,y+1) if i%n==0])


# A python function that take an integer and returns the list of all positive divisors of that integer.
def all_divs(n):
    result = []
    if n<=0:
        return result
    return [i for i in range(1,n+1) if n%i==0]


# A python function that take an integer and returns the sum of all positive divisors of that integer.
def sum_all_divs(n):
    result = []
    if n<=0:
        return sum(result)
    return sum([i for i in range(1,n+1) if n%i==0])


# A python function that returns the list of all prime numbers up to n inclusive.
def prime_nums(n):
    result = []
    if n <= 1:
        return result
    else:
        flag = False
        for i in range(2, n + 1):
            for j in range(2, i):
                if i % j == 0:
                    flag = True
                    break
            if flag:
                flag = False
            else:
                result.append(i)
        return result
                

# A python function that returns the sum of all prime numbers up to n inclusive.
def sum_prime_nums(n):
    result = []
    if n <= 1:
        return sum(result)
    else:
        flag = False
        for i in range(2, n + 1):
            for j in range(2, i):
                if i % j == 0:
                    flag = True
                    break
            if flag:
                flag = False
            else:
                result.append(i)
        return sum(result) 


# A python function that returns the list of all prime numbers between x and y both inclusive.
def prime_nums_x_y(x,y):
    result = []
    if x > y:
        return result
    elif x <= 1 and y <= 1:
        return result
    elif x <= 1:
        return prime_nums_x_y(2,y)
    else:
        flag = False
        for i in range(x, y + 1):
            for j in range(2, i):
                if i % j == 0:
                    flag = True
                    break
            if flag:
                flag = False
            else:
                result.append(i)
        return result


# A python function that returns the sum of all prime numbers between x and y both inclusive.
def sum_prime_nums_x_y(x,y):
    result = []
    if x > y:
        return sum(result)
    elif x <= 1 and y <= 1:
        return sum(result)
    elif x <= 1:
        return sum_prime_nums_x_y(2,y)
    else:
        flag = False
        for i in range(x, y + 1):
            for j in range(2, i):
                if i % j == 0:
                    flag = True
                    break
            if flag:
                flag = False
            else:
                result.append(i)
        return sum(result)


# A python function that returns the greatest common factor of x and y.
def gcf(x, y):
    result = None
    if x == 0 and y == 0:
        return result
    elif x < 0:
        return gcf(-x, y)
    elif y < 0:
        return gcf(x, -y)
    elif x == 0:
        return y
    elif y == 0:
        return x
    else:
        x_factors = [i for i in range(1, x + 1) if x % i == 0]
        y_factors = [i for i in range(1, y + 1) if y % i == 0]
        for i in reversed(x_factors):
            if i in y_factors:
                result = i
                break
        return result


# A python function that returns the greatest common factor of x and y and z.
def gcf(x, y, z):
    result = None
    if x == 0 and y == 0 and z == 0:
        return result
    elif x < 0:
        return gcf(-x, y, z)
    elif y < 0:
        return gcf(x, -y, z)
    elif z < 0:
        return gcf(x, y, -z)
    else:
        if x == 0:
            x_factors = [i for i in range(0, max(y, z) + 1)]
        else:
            x_factors = [i for i in range(1, x + 1) if x % i == 0]
        if y == 0:
            y_factors = [i for i in range(0, max(x, z) + 1)]
        else:
            y_factors = [i for i in range(1, y + 1) if y % i == 0]
        if z == 0:
            z_factors = [i for i in range(0, max(x, y) + 1)]
        else:
            z_factors = [i for i in range(1, z + 1) if z % i == 0]

        for i in reversed(x_factors):
            if i in y_factors and i in z_factors:
                result = i
                break
        return result


# A python function that takes a positive integer and returns true if the integer is a perfect number otherwise it should return false.
def is_perfect_number(n):
    if n<=0:
        return False
    return sum([i for i in range(1, n) if n % i == 0]) == n


# A python function that returns the list of all perfect numbers up to x inclusive.
def all_perfect_numbers(x):
    result = []
    if x<=0:
        return result
    for n in range(1,x+1):
        if sum([i for i in range(1, n) if n % i == 0]) == n:
            result.append(n)
    return result