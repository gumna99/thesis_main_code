#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
 
class SAX_trans:
    
    def __init__(self, ts, w, alpha):
        self.ts = ts
        self.w = w
        self.alpha = alpha
        self.aOffset = ord('a') #字符的起始位置，从a开始
        self.breakpoints = {'3' : [-0.43, 0.43],
                            '4' : [-0.67, 0, 0.67],
                            '5' : [-0.84, -0.25, 0.25, 0.84],
                            '6' : [-0.97, -0.43, 0, 0.43, 0.97],
                            '7' : [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
                            '8' : [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
            
        }
        self.beta = self.breakpoints[str(self.alpha)]
        
#     def normalize(self):  # 正则化
#         X = np.asanyarray(self.ts)
#         return (X - np.nanmean(X)) / np.nanstd(X)
 
    def paa_trans(self):  #转换成paa
        tsn = self.ts 
        paa_ts = []
        n = len(tsn)
        xk = math.ceil( n / self.w )  #math.ceil()上取整，int()下取整
        for i in range(0,n,xk):
            temp_ts = tsn[i:i+xk]
            paa_ts.append(np.mean(temp_ts))
            i = i + xk
        return paa_ts
    
    def to_sax(self):   #转换成sax的字符串表示
        tsn = self.paa_trans()
        len_tsn = len(tsn)
        len_beta = len(self.beta)
        strx = ''
        for i in range(len_tsn):
            letter_found = False
            for j in range(len_beta):
                if np.isnan(tsn[i]):
                    strx += '-'
                    letter_found = True
                    break                   
                if tsn[i] < self.beta[j]:
                    strx += chr(self.aOffset +j)
                    letter_found = True
                    break
            if not letter_found:
                strx += chr(self.aOffset + len_beta)
        return strx
    def to_sax2(self):   #转换成sax的字符串表示
        tsn = self.paa_trans()
        len_tsn = len(tsn)
        len_beta = len(self.beta)
        strx = ''
        for i in range(len_tsn):
            letter_found = False
            for j in range(len_beta):
                if np.isnan(tsn[i]):
                    strx += '-'
                    letter_found = True
                    break                   
                if tsn[i] < self.beta[j]:
                    jj=j
                    strx += np.str(jj)
                    letter_found = True
                    break
            if not letter_found:
                strx += np.str(len_beta)
        return strx
    
    def compare_Dict(self):   # 生成距离表 
        num_rep = range(self.alpha)  #存放下标
        letters = [chr(x + self.aOffset) for x in num_rep]   #根据alpha，确定字母的范围
        compareDict = {}
        len_letters = len(letters)
        for i in range(len_letters):
            for j in range(len_letters):
                if np.abs(num_rep[i] - num_rep[j])<=1:
                    compareDict[letters[i]+letters[j]]=0
                else:
                    high_num = np.max([num_rep[i], num_rep[j]])-1
                    low_num = np.min([num_rep[i], num_rep[j]])
                    compareDict[letters[i]+letters[j]] = self.beta[high_num] - self.beta[low_num]
        return compareDict
   
    def dist(self, strx1,strx2):   #求出两个字符串之间的mindist()距离值
        len_strx1 = len(strx1)
        len_strx2 = len(strx2)
        com_dict = self.compare_Dict()
 
        if len_strx1 != len_strx2:
            print("The length of the two strings does not match")
        else:
            list_letter_strx1 = [x for x in strx1]
            list_letter_strx2 = [x for x in strx2]
            mindist = 0.0
            for i in range(len_strx1):
                if list_letter_strx1[i] is not '-' and list_letter_strx2[i] is not '-':
                    mindist += (com_dict[list_letter_strx1[i] + list_letter_strx2[i]])**2
            mindist = np.sqrt((len(self.ts)*1.0)/ (self.w*1.0)) * np.sqrt(mindist)
            return mindist
        
class sax2_to_score:
    # 原partF 的公式
    def code(s,rank):
        code_list=[]
        for i in range(len(s)-2):
            code_list.append(s[i:i+3])
        c_appear = dict((a, code_list.count(a)) for a in code_list) #計數
        c_a_r = sorted(c_appear.items(),key =lambda x : x[1],reverse=True)
        code_CR = c_a_r[:rank]
        total_sum = 0 
        for i in code_CR:
            total_sum += i[1]
        print(total_sum)
        print(code_CR)
        return code_CR,total_sum

    def score(code_CR,total_sum):
        score = 0
        for i in code_CR:
            code = i[0]
            count= i[1]
            freq = count/total_sum

            code_int =[]
            for i in range(3):
                code_int.append(int(code[i]))

            score_f = freq * ( abs(np.mean(code_int) - 3) + 0.5 * np.std(code_int))
            score += score_f

        return score
            
    def score_3s(s):
        score = []
        for i in s:
            code_int = list(map(int, str(i)))
            score_code = ( abs(np.mean(code_int) - 3) + 0.5 * np.std(code_int))
            score.append(score_code)
        return score


# In[ ]:




