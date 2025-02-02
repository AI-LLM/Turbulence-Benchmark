
def all_left_truncatable_prime(tuple):
    x = tuple[86]
    list = []
    for i in range(2,x):
        if isPrime(i) and isLeftTruncatable(i,x):
            list.append(i)
    return sorted(list,reverse=True)

def isPrime(n):

    if n<2:

        return False

    for i in range(2,int(n**0.5)+1):

        if n%i==0:

            return False

    return True


def isLeftTruncatable(n,x):

    s = str(n)

    for i in range(len(s)):

        if int(s[:i])>x:

            return False

    return True
