import time


# def greeting(expr):
#     # Normal wrapper
#     def greeting_decorator(func):
#         def function_wrapper(x):
#             print(expr + ", " + func.__name__ + " returns:")
#             func(x)
#         return function_wrapper
#     return greeting_decorator
#
# @greeting("καλημερα")
# def foo(x):
#     print(42)
#
# foo("Hi")
# καλημερα, foo returns:
# 42


def time_this_function(func):
    def wrapper_func(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end=time.time()

        return end-start
    return wrapper_func


def print_name_and_time(func):
    def wrapper_func(*args, **kwargs):
        print(func.__name__, end='\t')
        start = time.time()
        func(*args, **kwargs)
        end=time.time()

        return end - start
    return wrapper_func


