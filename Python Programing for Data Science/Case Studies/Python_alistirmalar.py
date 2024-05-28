#Gamze Öğütcü - Miuul Python Alıştırmalar


# Görev 1: Verilen değerlerin veri yapılarını inceleyiniz.
x = 8  # int
type(x)

y = 3.2  # float
type(y)

z = 8j + 18  # complex
type(z)

a = 'Hello World'  # string
type(a)

b = True  # bool
type(b)

c = 23 < 22  # bool
type(c)

l = [1, 2, 3, 4]  # list
type(l)

d = {"Name": "Jake",
     "Age": 27,
     "Address": "Downtown"}  # dictionary
type(d)

t = ("Machine Learning", "Data Science")  # tuple
type(t)

s = {'Python', 'Machine Learning', 'Data Science'}  # set
type(s)


# Görev 2:Verilen string ifadenin tüm harflerini büyük harfe çeviriniz.
# Virgül ve nokta yerine space koyunuz.
# Kelime kelime ayırınız.
text = "The goal is to turn data into information, and information into insight"

text.upper().replace(",", " ").replace(".", " ").split()



# Görev 3: Verilen listeye aşağıdaki adımları uygulayınız.
lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
#Adım 1: Verilen listenin eleman sayısına bakınız.
len(lst)

#Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
lst[0]
lst[10]

# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
lst[0:4]

# Adım 4: Sekizinci indeksteki elemanı siliniz.
lst.pop(8)

# Adım 5: Yeni bir eleman ekleyiniz.
lst.append("G")

# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
lst.insert(8, "N")




#Görev 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

#Adım 1: Key değerlerine erişiniz.
dict.keys()

#Adım 2: Value'lara erişiniz.
dict.values()

# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict["Daisy"][1] = 13

# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dict['Ahmet'] = ["Turkey", 24]

# Adım 5: Antonio'yu dictionary'den siliniz.
dict.pop("Antonio")


# Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan
# ve bu listeleri return eden fonksiyon yazınız.
def odd_even_list(l):
    odd_list = []
    even_list = []
    [even_list.append(i) if i % 2 == 0 else odd_list.append(i) for i in l]
    return even_list, odd_list


even_list,odd_list=odd_even_list([2, 13, 18, 93, 22])


# Görev 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrencide tıp fakültesi
# öğrenci sırasına aittir. Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
def successful_students(ogrenciler):
    for index, ogrenci in enumerate(ogrenciler, 1):
        if index < 4:
            print("Mühendislik Fakültesi " + str(index) + " . öğrenci: " + ogrenci)
        else:
            index-=3
            print("Tıp Fakültesi " + str(index) + " . öğrenci: " + ogrenci)


successful_students(["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"])

# Görev 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri
# yer almaktadır. Zip kullanarak ders bilgilerini bastırınız.
ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

zipped_info = zip(kredi, ders_kodu, kontenjan)
type(zipped_info)
#1.yol
for i in zipped_info:
    print("Kredisi " + str(i[0]) + " olan ", i[1], " kodlu dersin kontenjanı ", str(i[2]), " kişidir.")

#2.yol
for ders_kodu, kredi, kontenjan in zipped_info:
  print(f"Kredisi {kredi} olan {ders_kodu} kodlu dersin kontenjanı {kontenjan} kişidir.")

# Görev 8: Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını
# eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.
kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])


def kume(set1, set2):
    if set1.issuperset(set2):
        print(set1.intersection(set2))
    else:
        print(set2.difference(set1))


kume(kume1, kume2)
