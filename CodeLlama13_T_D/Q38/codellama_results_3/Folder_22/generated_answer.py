
def find_subset_of_length_n(elements, n):
    return len([s for s in powerset(elements) if len(s) == n])
```
Note: The `powerset` function is defined as follows:
```
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
