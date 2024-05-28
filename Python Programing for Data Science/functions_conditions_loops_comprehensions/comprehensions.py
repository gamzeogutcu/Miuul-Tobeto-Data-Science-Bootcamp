salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(x):
    return x * 20 / 100 + x


null_list = []

for salary in salaries:
    null_list.append((new_salary(salary)))

for salary in salaries:
    if salary > 3000:
        null_list.append((new_salary(salary)))
    else:
        null_list.append(new_salary(salary * 2))

[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]

[salary * 2 for salary in salaries]
[salary * 2 for salary in salaries if salary < 3000]
[salary * 2 if salary < 3000 else salary * 0 for salary in salaries]
[new_salary(salary * 2) if salary < 3000 else new_salary(salary * 0.2) for salary in salaries]

students = ["John", "Mark", "Venessa", "Mariam"]
students_no = ["John", "Venessa"]

[student.lower() if student in students_no else student.upper() for student in students]
"John".lower()

dictionary = {'a': 1,
              'b': 2,
              'c': 3,
              'd': 4}

dictionary.keys()
dictionary.values()
dictionary.items()

{k: v ** 2 for (k, v) in dictionary.items()}
{k.upper(): v for (k, v) in dictionary.items()}
{k.upper(): v ** 2 for (k, v) in dictionary.items()}

numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

{n: n ** 2 for n in numbers if n % 2 == 0}

#############################
# Bir veri setindeki değişken isimlerini değiştirme
##########

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns
A = []
for col in df.columns:
    A.append(col.upper())

df.columns = A

df = sns.load_dataset("car_crashes")
df.columns = [col.upper() for col in df.columns]

[col for col in df.columns if "INS" in col]
["FLAG_"+col for col in df.columns if "INS" in col]

["FLAG_"+col if "INS" in col else "NO_FLAG_"+col for col in df.columns]
df.columns=["FLAG_"+col if "INS" in col else "NO_FLAG_"+col for col in df.columns]

############

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

num_cols=[col for col in df.columns if df[col].dtype != "O"]
soz={}
agg_list=["mean","min","max","sum"]

for col in num_cols:
    soz[col]=agg_list

new_dict = {col:agg_list for col in num_cols}
df[num_cols].head()
df[num_cols].agg(new_dict)