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

# Using dictionaries for key-value pairs
student = {
    'name': 'John',
    'age': 20,
    'grades': [85, 90, 88],
    'is_active': True
}

# Accessing dictionary values
print("\nDictionary Examples:")
print("Student name:", student['name'])
print("Student age:", student.get('age'))  # Using get() method (safer)

# Adding and modifying dictionary entries
student['major'] = 'Computer Science'  # Add new key-value pair
student['age'] = 21  # Modify existing value
print("Updated student info:", student)

# Dictionary methods
print("\nDictionary Methods:")
print("Keys:", student.keys())
print("Values:", student.values())
print("Items:", student.items())

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
print("\nSquares dictionary:", squares)

# Nested dictionaries
classroom = {
    'students': {
        'john': {'grade': 85, 'attendance': 90},
        'alice': {'grade': 92, 'attendance': 95}
    },
    'teacher': {
        'name': 'Dr. Smith',
        'subject': 'Mathematics'
    }
}
print("\nNested dictionary example:", classroom)

# Accessing nested dictionary values
johns_grade = classroom['students']['john']['grade']
print("\nJohn's grade:", johns_grade)

# Practical Examples of Dictionary Usefulness

# 1. Counting frequencies (very common use case)
text = "hello world hello python world"
word_count = {}
for word in text.split():
    word_count[word] = word_count.get(word, 0) + 1
print("\nWord frequency:", word_count)

# 2. Caching/Memoization (storing computed results)
fib_cache = {}
def fibonacci(n):
    if n in fib_cache:
        return fib_cache[n]
    if n <= 1:
        return n
    fib_cache[n] = fibonacci(n-1) + fibonacci(n-2)
    return fib_cache[n]
print("\nFibonacci with caching:", fibonacci(10))

# 3. Configuration Settings
config = {
    'database': {
        'host': 'localhost',
        'port': 5432,
        'username': 'admin',
        'password': 'secret'
    },
    'api': {
        'endpoint': 'https://api.example.com',
        'timeout': 30
    }
}
print("\nConfiguration:", config)

# 4. Graph Representation
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
print("\nGraph representation:", graph)

# 5. User Session Management
user_sessions = {
    'user123': {
        'last_login': '2024-03-20',
        'cart_items': ['item1', 'item2'],
        'preferences': {'theme': 'dark', 'language': 'en'}
    },
    'user456': {
        'last_login': '2024-03-19',
        'cart_items': ['item3'],
        'preferences': {'theme': 'light', 'language': 'es'}
    }
}
print("\nUser sessions:", user_sessions)

# 6. API Response Handling
api_response = {
    'status': 'success',
    'data': {
        'products': [
            {'id': 1, 'name': 'Laptop', 'price': 999.99},
            {'id': 2, 'name': 'Phone', 'price': 499.99}
        ],
        'total_items': 2
    },
    'message': 'Products retrieved successfully'
}
print("\nAPI response:", api_response)

