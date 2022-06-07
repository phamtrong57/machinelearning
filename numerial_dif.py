import sys

#numerial diff with h ~ 1e50
def numerial_diff(f,x):
    h = 1e-4
    return (f(x + h)-f(x)) / h

def f(x):
    return 0.01*x**2 + 0.1*x

if __name__ == "__main__":
    x = sys.argv[1]
    print(numerial_diff(f,int(x)))