
import re

imgname = "02Conv-LSTM2000_gt01.png"
list1 = re.findall("\d",imgname)
list2 = re.findall("(?<=\d\d)\D+",imgname) #*(?=\d)
print("list1 = ",list1)
print("list2 = ",list2)
str1 = "".join(list1[2:-2])
str2 = "".join(list2[0])

print("str1 = ",str1)
print("str2 = ",str2)