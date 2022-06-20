import sys

def get_argv(array_of_element_numbers):
    result = [sys.argv[i] for i in array_of_element_numbers]
    return result