
def if_perfect_num(mylist):
    if mylist[985] % sum(mylist[985:]) == 0:
        return True
    else:
        return False
