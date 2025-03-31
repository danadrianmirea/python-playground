from testfunc import func1, func2, func3

# Method 1: Using a dictionary
function_map = {
    'func1': func1,
    'func2': func2,
    'func3': func3
}

# Method 2: Using a list
functions = [func1, func2, func3]

# Method 3: Using getattr
import testfunc

def demo_all_methods():
    print("Method 1 - Using dictionary:")
    function_map['func1']()
    function_map['func2']()
    function_map['func3']()
    
    print("\nMethod 2 - Using list:")
    functions[0]()
    functions[1]()
    functions[2]()
    
    print("\nMethod 3 - Using getattr:")
    getattr(testfunc, 'func1')()
    getattr(testfunc, 'func2')()
    getattr(testfunc, 'func3')()

if __name__ == "__main__":
    demo_all_methods() 