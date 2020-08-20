import jieba
import numpy as np
import pandas as pd
delat = [[0 for i in range(4)] for i in range(3)]
print(delat)
for i in range(4):
    delat[0][i] +=1
print(delat)
key_value = {}
df1=[]
df1 = pd.DataFrame(df1)

df = pd.DataFrame([['A', 1, 2,3,4], ['B', 5, 6,7,8],['C',1,2,3,4]],  columns=['word', 'c1', 'c2','c3','tfidf'])
df2 = pd.DataFrame([['B', 1, 2,3,40], ['C', 5, 6,7,80],['D',1,2,3,40]],  columns=['word', 'c1', 'c2','c3','tfidf'])



a = pd.DataFrame(df[['word','tfidf']].set_index('word')).T
b = pd.DataFrame(df2[['word','tfidf']].set_index('word')).T
print("aaaaaaaaaa")
print(a)
print("bbbbbbb")
print(b)
c = pd.concat([a,b],axis=0,ignore_index=True,sort=False, join='outer')
print("cccccccc")
print(c)
# c = np.array(a)
# print(c)
# print(c.shape)

# sum=[]
# k=3
# for i in range(k):
#     sum[i]=1
# print(sum)
#
# 初始化
key_value[2] = 56
key_value[1] = 2
key_value[5] = 12
key_value[4] = 24
key_value[6] = 18
key_value[3] = 323
#
list1=sorted(key_value.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

list2=pd.DataFrame(list1)
list3 = np.array(list1)
list2.columns=['a','b']
# listc=[1,2,3,4,5,6]
# list2['c']=listc
list4 = np.array(list2).tolist()
print(len(list4))

# print(list4[:,0])
#
# cluster = np.mat(np.zeros((3, 2)))
#
# print(cluster[:,0].A)


# list ={"1","2","3"}
# l=len(list)
#
# s=np.array([i for i in list])
# t=np.zeros(l)
# s=np.c_[s,t]
# print(s)
#
# print(sorted(s, key=lambda a: a[0]))



#
# strs="我 爱 中 国"
# dicts={}
# strs=strs.replace(" ","")
# print(strs)
# stopwords = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8').readlines()]
# all_seg_list = jieba.cut(strs, cut_all=False) # 精确模式
# i=0
# for word in all_seg_list:
#     if word not in stopwords:
#       if not word in dicts:
#             i=i+1
#             dicts[word]=i
# print(dicts)
# single_seg_list = jieba.cut(strs, cut_all=False)
# for word in single_seg_list:
#     if word in dicts:
#         print(word,dicts[word])
#         strs=strs.replace(word,str(dicts[word]))
#
#
# print("____________________________________________________")
# print("writelines" + strs)
# # print("write" + data)
# a="我 爱 中 国"
# # print(a.replace("我","1"))
# # b="w as s"
# # print(b.replace("w","s"))
