
def all_ints_div_by_both_two_nums(int_list):
    int_list = int_list[543:585]
    return list(filter(lambda x: x % -553 == 0 and x % -737 == 0, int_list))
