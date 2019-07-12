#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
 
class Score():
    def __init__(self,pre_score,rel_label,threshold,beta):
        self.tn = 0
        self.fn = 0
        self.fp = 0
        self.tp = 0
        self.pre_score = pre_score
        self.rel_label = rel_label
        self.threshold = threshold
        self.beta = beta
        list(map(self.__getCM_count,
                 self.pre_score,
                 self.rel_label))
 
    def __getCM(self,pre, rel):
        if (pre < self.threshold):
            if (rel == 0): return 'TN'
            if (rel == 1): return 'FN'
        if (pre >=  self.threshold):
            if (rel == 0): return 'FP'
            if (rel == 1): return 'TP'
 
    def get_cm(self):
        return list(map(self.__getCM,
                        self.pre_score,
                        self.rel_label))
 
    def __getCM_count(self,pre, rel):
        if (pre < self.threshold):
            if (rel == 0): self.tn += 1
            if (rel == 1): self.fn += 1
        if (pre >=  self.threshold):
            if (rel == 0): self.fp += 1
            if (rel == 1): self.tp += 1
 
    def get_f1(self):
        P = self.tp/(self.tp+self.fp)
        R = self.tp/(self.tp+self.fn)
        if(P == 0.0):
            return 0.0
        else:
            return (self.beta*self.beta+1)*P*R/(self.beta*self.beta*P+R)
 
    # 方法二 precision——分数精度
    def get_auc_by_count(self,precision=100):
        # 正样本数
        postive_len = sum(self.rel_label)
        # 负样本数
        negative_len = len(self.rel_label) - postive_len
        # 总对比数
        total_case = postive_len * negative_len
        # 正样本分数计数器(填0在range...)
        pos_histogram = [0 for _ in range(precision+1)]
        # 负样本分数计数器(填0在range...)
        neg_histogram = [0 for _ in range(precision+1)]
        # 分数放大
        bin_width = 1.0 / precision
 
        for i in range(len(self.rel_label)):
            nth_bin = int(self.pre_score[i] / bin_width)
            if self.rel_label[i] == 1:
                pos_histogram[nth_bin] += 1
            else:
                neg_histogram[nth_bin] += 1
 
        accumulated_neg = 0
        satisfied_pair = 0
        for i in range(precision+1):
            satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
            accumulated_neg += neg_histogram[i]
        return satisfied_pair / float(total_case)
 
    # 方法三
    def get_auc_by_rank(self):
        # 拼接排序
        df = pd.DataFrame({'pre_score':self.pre_score,'rel_label':self.rel_label})
        df = df.sort_values(by='pre_score',ascending=False).reset_index(drop=True)
        # 获取 n，N，M
        n = len(df)
        M = len(df[df['rel_label']==1])
        N = n - M
        # 初始化rank 和同值统计ank_tmp,count_all,count_p
        rank = 0.0
        rank_tmp,count_all,count_p = 0.0,0,0
        # 添加防止越界的一条不影响结果的记录
        df.loc[n] = [0,0]
        # 遍历一次
        for i in range(n):
            # 判断i+1是否与i同值，不同值则要考虑是否刚刚结束同值统计
            if(df['pre_score'][i+1] != df['pre_score'][i]):
                # 正样本
                if(df['rel_label'][i] == 1):
                    # 计数不为0，刚刚结束同值统计
                    if (count_all != 0):
                        # 同值统计结果加在rank上，这里注意补回结束统计时漏掉的最后一条同值数据
                        rank += (rank_tmp + n - i) * (count_p+1) / (count_all+1)
                        rank_tmp, count_all, count_p = 0.0, 0, 0
                        continue
                    rank += (n-i)
                else:
                    if (count_all != 0):
                        rank += (rank_tmp + n - i) * (count_p) / (count_all+1)
                        rank_tmp, count_all, count_p = 0.0, 0, 0
                        continue
            else:
                rank_tmp += (n-i)
                count_all += 1
                if(df['rel_label'][i] == 1):
                    count_p += 1
        return (rank-M*(1+M)/2)/(M*N)
 
 
if __name__ == '__main__':
    learn_data_L2 = [0.2,0.3,0.4,0.35,0.6,0.55,0.2,0.57,0.3,0.15,0.77,0.33,0.9,0.49, 0.45,0.41, 0.66,0.43,0.7,0.4]
    learn_data_R2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    learn_data2 = pd.DataFrame({'Learn': learn_data_L2, 'Real': learn_data_R2})
 
    score2 = Score(learn_data2['Learn'], learn_data2['Real'], 0.5, 1)
 
    print(score2.get_cm())
    print(score2.get_f1())
    print(score2.get_auc_by_count())
    print(score2.get_auc_by_rank())


# In[ ]:




