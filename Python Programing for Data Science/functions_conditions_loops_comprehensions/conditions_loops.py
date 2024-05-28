#############

if 1 == 1:
    print("something")

if 1 == 2:
    print("something")

number = 11


def number_check(number):
    if number == 10:
        print("number is 10")

    else:
        print("number is not 10")


number_check(11)


def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("number is 10")


number_check(8)

#####################33
students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student)

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)


def new_salary(salary, rate):
    return int(salary * rate / 100 + salary)


for salary in salaries:
    print(new_salary(salary, 20))

for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))


######################33
def alternating(str):
    new_str = ""
    for i in range(len(str)):
        if i % 2 == 1:
            new_str += str[i].lower()
        else:
            new_str += str[i].upper()
    return new_str


alternating("Gamze")

#####################3
salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        break
    print(salary)

for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

number = 1
while number < 5:
    print(number)
    number += 1

students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student)


def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    return groups


divide_students(students)


def alternating_with_enumarate(string):
    new_string = ""

    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()

    print(new_string)


alternating_with_enumarate("Hi my name is gamze")

#########################333
departments = ["mathematics", "statistics", "physics", "astronomy"]
ages = [23, 30, 2, 22]
list(zip(students, departments, ages))


###################

def summer(a, b):
    return a + b


summer(1, 3) * 9

new_sum = lambda a, b: a + b
new_sum(4, 5)

salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(x):
    return x * 20 / 100 + x


new_salary(5000)

for salary in salaries:
    print(new_salary(salary))

list(map(lambda x: x * 20 / 100 + x, salaries))


city_name =["London","Paris","Berlin"]

def plate(cities):
    for index,city in enumerate(cities,1):
        print(f"{index}: {city}")

plate(city_name)