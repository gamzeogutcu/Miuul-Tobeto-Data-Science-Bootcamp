import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []
for i in range(0, len(a)):
    ab.append(a[i] * b[i])

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype=int)
np.random.randint(0, 10, size=10)
np.random.normal(10, 4, (3, 4))

a = np.random.randint(10, size=5)
a.ndim
a.shape
a.size
a.dtype

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)

# index
a = np.random.randint(10, size=10)
a[0]
a[0:5]
a[0] = 999

m = np.random.randint(10, size=(3, 5))
m[0, 0]
m[1, 1]
m[2, 3] = 2.9
m[:, 0]
m[1, :]
m[0:2, 0:3]

# fancy index

import numpy as np

v = np.arange(0, 30, 3)
v[1]
v[4]

catch = [1, 2, 3]
v[catch]

###numpy koşullu işlemler
v = np.array([1, 2, 3, 4, 5])

ab = []
for i in v:
    if i < 3:
        ab.append(i)

v[v < 3]
v[v > 3]
v[v != 3]

#####3Matematiksel işlemler
v / 5
v * 5 / 10
v ** 2
v - 1
np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)
v = np.subtract(v, 1)

# 5*x0+x1=12
# x0+3*x1=10
a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])
np.linalg.solve(a, b)

#################################PANDAS
import pandas as pd

s = pd.Series([19, 77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(3)
s.tail(3)

# Veri Okuma
df = pd.read_csv("datasets/advertising.csv")
df.head()

import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["sex"].value_counts()

# Selection in PAnda
df.index
df[0:13]
df.drop(0, axis=0).head()

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)

# DEğişkeni indexe çevirmek
df["age"].head()
df.age.head()
df.index = df["age"]
df.drop("age", axis=1).head()

df.drop("age", axis=1, inplace=True)
df.head()

# indexi değişkene çevirmek
df.index
df["age"] = df.index
df.head()

df = df.reset_index().head()

pd.set_option("display.max_columns", None)

"age" in df
df["age"].head()
df.age.head()

type(df["age"].head())

df[["age"]].head()
type(df[["age"]].head())
df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"] ** 2
df["age3"] = df["age"] / df["age2"]
df.drop("age3", axis=1).head()
df.drop(col_names, axis=1).head()
df.loc[:, ~df.columns.str.contains("age")].head()

df.head()

# iloc and loc
df.iloc[0:3]
df.iloc[0, 0]
df.iloc[0:3, 0:3]

# loc label based selection
df.loc[0:3]
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]

# koşullu seçim
df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()
df.loc[df["age"] > 50, ["age", "class"]].head()
df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head()
df_new = df.loc[(df["age"] > 50)
                & (df["sex"] == "male")
                & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
["age", "class", "embark_town"]]
df_new["embark_town"].value_counts()

###############toplulaştırma ve gruplama
df["age"].mean()
df.groupby("sex")["age"].mean()
df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean", "sum"]})
df.groupby("sex").agg({"age": ["mean", "sum"], "survived": "mean"})
df.groupby(["sex", "embark_town"]).agg({"age": ["mean"], "survived": "mean"})
df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"], "survived": "mean"})
df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"], "survived": "mean", "sex": "count"})

####### Pivot table
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.pivot_table("survived", "sex", "embarked")
df.pivot_table("survived", "sex", "embarked", aggfunc="std")
df.pivot_table("survived", "sex", ["embarked", "class"])

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
df.pivot_table("survived", "sex", "new_age")
df.pivot_table("survived", "sex", ["new_age", "class"])
df.dtypes
pd.set_option("display.width", 500)

############Apply ve Lambda
df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5

(df["age"] / 10).head()
(df["age2"] / 10).head()
(df["age3"] / 10).head()

for col in df.columns:
    if "age" in col:
        print((df[col] / 10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col] / 10

df.head()

df[["age", "age2", "age3"]].apply(lambda x: x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()


def standard_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()


df.loc[:, df.columns.str.contains("age")].apply(standard_scaler).head()

df.loc[:, ["age", "age2", "age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standard_scaler).head()
df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standard_scaler).head()
df.head()

####3Birleştirme
import numpy as np

m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2], ignore_index=True)

df1 = pd.DataFrame({'employees': ["john", 'dennis', 'mark', 'maria'],
                    'group': ["accounting", "engineering", "engineering", "hr"]})
df2 = pd.DataFrame({'employees': ['mark', "john", 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})

df3 = pd.merge(df1, df2)
pd.merge(df1, df2, on="employees")

df4 = pd.DataFrame({'group': ["accounting", "engineering", "hr"],
                    'manager': ["Caner", 'Mustafa', 'Berkcan'], })
pd.merge(df3, df4)
dict = {"Paris": [10], "Berlin": [20]}
pd.DataFrame(dict)

############## VERİ GÖRSELLEŞTİRME: MATPLOTLIB SEABORN
# Kategorik: sütun grafik,countplot bar
# Sayısal değişken: hist, boxplot


####Kategorik Değişken Görselleştirme
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

df["sex"].value_counts().plot(kind='bar')
plt.show()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])

#####Matplotlib özellikleri
# plot
x = np.array([1, 8])
y = np.array([0, 150])
plt.plot(x, y, 'o')

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])
plt.plot(x, y, 'o')

plt.plot(y, linestyle="dotted", color='r')
plt.plot(x)
plt.plot(y)

# labels
plt.title("bu ana başlık")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

# subplot
# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 1)
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 3, 2)
plt.title("2")
plt.plot(x, y)

# plot 3
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 3)
plt.title("3")
plt.plot(x, y)

# seaborn
df = sns.load_dataset("tips")
df.head()
df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

df["sex"].value_counts().plot(kind='bar')

# sayısal değişken
sns.boxplot(x=df["total_bill"])
sns.boxplot()
(df["total_bill"]).hist()
plt.show()
########Keşifçi Veri Analizi

###################3Genel Resim
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()


def check_df(dataframe, head=5):
    print("##### Shape #####")
    print(dataframe.shape)
    print("##### Types #####")
    print(dataframe.dtypes)
    print("##### Head #####")
    print(dataframe.head())
    print("##### Tail #####")
    print(dataframe.tail())
    print("##### NA #####")
    print(dataframe.isnull().sum())
    print("##### Quantiles #####")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

df = sns.load_dataset("flights")
check_df(df)

##########2) Kategorik Değişken Analizi(Analysis of Categorical Variables)
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

df["embarked"].value_counts()
df["sex"].unique()
df["class"].nunique()
num_cols = df.select_dtypes("number").columns.to_list()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
# num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and col in df.select_dtypes("number").columns.to_list()]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("####################################")


cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("####################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


cat_summary(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("sdfksıdsld")
    else:
        cat_summary(df, col, plot=True)

df["adult_male"].astype("int64")

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype("int64")
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)


def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("####################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("####################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


cat_summary(df, "adult_male", plot=True)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("####################################")


cat_summary(df, "sex")

###################3.Sayısal Değişken Analizi
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[["age", "fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float"]]
num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe, numerical_cols):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_cols].describe(quantiles).T)


num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)


def num_summary(dataframe, numerical_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_cols].describe(quantiles).T)

    if plot:
        dataframe[numerical_cols].hist()
        plt.xlabel(numerical_cols)
        plt.title(numerical_cols)
        plt.show(block=True)


num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

############## Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
  Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin ismini verir.

  Parameters
  ----------
  dataframe: dataframe
      değişken isimleri alınmak istenen dataframe'dir.
  cat_th: int, float
      numerik fakat kategorik olan değişkenler için sınıf eşik değeri
  car_th: int, float
      kategorik fakat kardinal değişkenler için sınıf eşik değeri
  Returns
  -------
  cat_cols: list
      Kategorik değişkenlerin listesi
  num_cols: list
      Numerik değişken listesi
  cat_but_car: list
      Kategorik görünümlü kardinal değişken listesi
  Notes
  -------
  cat_cols + num_cols + cat_but_car = toplam değişken sayısı
  num_but_cat cat_cols'un içerisinde.
  """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int64", "float"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > car_th and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("####################################")


cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_cols].describe(quantiles).T)

    if plot:
        dataframe[numerical_cols].hist()
        plt.xlabel(numerical_cols)
        plt.title(numerical_cols)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)

### bonus
df = sns.load_dataset("titanic")
df.info()
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype("int64")

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("####################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

######## 4.Hedef Değişken Analizi
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype("int64")


def grab_col_names(dataframe, cat_th=10, car_th=30):
    """
  Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin ismini verir.

  Parameters
  ----------
  dataframe: dataframe
      değişken isimleri alınmak istenen dataframe'dir.
  cat_th: int, float
      numerik fakat kategorik olan değişkenler için sınıf eşik değeri
  car_th: int, float
      kategorik fakat kardinal değişkenler için sınıf eşik değeri
  Returns
  -------
  cat_cols: list
      Kategorik değişkenlerin listesi
  num_cols: list
      Numerik değişken listesi
  cat_but_car: list
      Kategorik görünümlü kardinal değişken listesi
  Notes
  -------
  cat_cols + num_cols + cat_but_car = toplam değişken sayısı
  num_but_cat cat_cols'un içerisinde.
  """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if
                   df[col].nunique() < 10 and col in df.select_dtypes("number").columns.to_list()]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.select_dtypes("number").columns.to_list() if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("####################################")

num_cols=df.select_dtypes("number").columns.to_list()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Hedef Değişkenin Kategorik Değişkenler ile Analizi

df.groupby("sex")["survived"].mean()
import seaborn as sns
import pandas as pd
df=sns.load_dataset("titanic")
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))


target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

## HEdef değişkenin Sayısal Değişkenler ile Analizi

df.groupby("survived")["age"].mean()
df.groupby("survived").agg({"age": "mean"})


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


target_summary_with_num(df, "survived", "age")


def num_summary(dataframe, numerical_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_cols].describe(quantiles).T)

    if plot:
        dataframe[numerical_cols].hist()
        plt.xlabel(numerical_cols)
        plt.title(numerical_cols)
        plt.show(block=True)


for col in num_cols:
    target_summary_with_num(df, "survived", col)

################### 5.Korelasyon Analizi

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

# Yüksek Korelasyonlu Değişkenlerin Silinmesi
cor_matrix = df.corr(numeric_only=True).abs()

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
cor_matrix[drop_list]
df = df.drop(drop_list, axis=1)

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr(numeric_only=True)
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df=df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)

#####Kaggle: train_transaction.csv
df = pd.read_csv("datasets/fraud_train_transaction.csv")


def cal(x, y):
    print(x * y)


cal(10, 40) - 200
df["sex"].describe([0.25, 0.50, 0.75])
titanic = sns.load_dataset("titanic")
sns.countplot(x="class", data=titanic)
df.select_dtypes("number").columns.to_list()
