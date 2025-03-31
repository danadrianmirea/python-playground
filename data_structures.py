v=[1,2,3,4,5]
v.append(6)
v.insert(0,3)
v.insert(2,0)
print(v)
a=v.pop()
print(a)
v.append(98)
print(v)

# Using deque for efficient first position insertions
from collections import deque
q = deque([1, 2, 3])
q.appendleft(0)  # O(1) operation to insert at beginning
print(q)
q.append(4)      # O(1) operation to insert at end
print(q)

for i in range(len(v)):
    print(v[i], end=' ')
print()  # Add a newline after the loop

for i in v:
    print(i, end=' ')
print()  # Add a newline after the loop

# Using sets for unique elements and set operations
numbers = {1, 2, 3, 3, 4, 4, 5}  # Duplicates are automatically removed
print("Set:", numbers)  # Will print: {1, 2, 3, 4, 5}

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

print("Union:", set1 | set2)      # {1, 2, 3, 4, 5, 6}
print("Intersection:", set1 & set2)  # {3, 4}
print("Difference:", set1 - set2)    # {1, 2}
print("Symmetric Difference:", set1 ^ set2)  # {1, 2, 5, 6}

# Adding and removing elements
numbers.add(6)      # Add a single element
numbers.update([7, 8, 9])  # Add multiple elements
print("After adding elements:", numbers)

try:
    numbers.remove(1)   # Remove an element (raises error if not found)
except KeyError:
    print("Element 1 not found in the set")
numbers.discard(2)  # Remove an element (no error if not found)
print("After removing elements:", numbers)

