print("a", "b")
print("a", "b", sep="__")


def calculate(x):
    print(x * 2)


calculate(5)


def summer(x, y):
    """
    Sum of two parameters

    :param x: int,float
    :param y: int,float
    :return: int,float
        x+y
    """
    print(x + y)


summer(8, 7)


def say_hi(string):
    print(string)
    print("Hi")
    print("Hello")


say_hi("Gamze")


def multiplication(a, b):
    c = a * b
    print(c)


multiplication(10, 9)

list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(1, 8)
add_element(18, 8)
add_element(180, 10)


###########################
def divide(a, b=1):
    print(a / b)


divide(2)


##########################33
def calculate(varm, moisture, charge):
    return (varm + moisture) / charge


calculate(98, 12, 78) * 10

a = calculate(98, 12, 78)


#############################
def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge
    return varm, moisture, charge, output


varm, moisture, charge, output = calculate(98, 12, 78)


############################

def calculate(varm, moisture, charge):
    return int((varm + moisture) / charge)


calculate(90, 12, 12) * 10


def standardization(a, p):
    return a * 10 / 100 * p * p


standardization(45, 1)


def all_calculation(varm, moisture, charge, p):
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)


all_calculation(1, 3, 5, 12)


def all_calculation(varm, moisture, charge, a, p):
    print(calculate(varm, moisture, charge))
    b = standardization(a, p)
    print(b * 10)


all_calculation(1, 3, 5, 19, 12)

###################################333
list_store = [1, 2]


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(1, 9)

store = []


def square(a, b):
    c = a ** b
    store.append(c)
    print(store)


square(4, 3)
