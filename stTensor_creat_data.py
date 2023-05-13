#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorly as tl
import random
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
import os
import math
from collections import Counter
import time
from scipy.spatial.distance import cdist
import networkx as nx
from sklearn.metrics import roc_auc_score
import random
import pylab
plt.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.metrics import f1_score
import scipy.stats as stats
import pyreadr
import heapq
import os


# In[88]:


spaCI1 = pyreadr.read_r('E://生物统计//代码//spaCI-main//src//spaCI_database.RDS')
spaCI1 = spaCI1[None]
spaCI = pd.DataFrame()
spaCI['Ligand'] = list(spaCI1['ligand'])
spaCI['Receptor'] = list(spaCI1['receptor'])
spaCI['LR'] = spaCI['Ligand'] + '-' + spaCI['Receptor']
# print(len(set(spaCI['ligand'])),len(set(spaCI['receptor'])))
# len(set(list(spaCI['ligand'])+list(spaCI['receptor'])))

iTALK1 = pyreadr.read_r('E://生物统计//代码//iTALK-master//iTALK-master//R//sysdata.rda')['database']
iTALK = pd.DataFrame()
iTALK['Ligand'] = list(iTALK1['Ligand.ApprovedSymbol'])
iTALK['Receptor'] = list(iTALK1['Receptor.ApprovedSymbol'])
iTALK['LR'] = iTALK['Ligand'] + '-' + iTALK['Receptor']

CellChatDB_human = pd.read_csv('D://zhuomian//by//data_create//CellChatDB_human.txt', sep=',')
CellChatDB_human_r1 = []
CellChatDB_human_r2 = []
for i in range(len(CellChatDB_human)):
    CellChatDB_human_r1.append(CellChatDB_human['receptor'][i].split('_')[0])
    CellChatDB_human_r2.append(
        CellChatDB_human['receptor'][i].split('_')[len(CellChatDB_human['receptor'][i].split('_')) - 1])
CellChatDB_human['Receptor1'] = CellChatDB_human_r1
CellChatDB_human['Receptor2'] = CellChatDB_human_r2
CellChatDB = pd.DataFrame()
CellChatDB['Ligand'] = list(CellChatDB_human['ligand']) + list(CellChatDB_human['ligand'])
CellChatDB['Receptor'] = CellChatDB_human_r1 + CellChatDB_human_r2
CellChatDB['LR'] = CellChatDB['Ligand'] + '-' + CellChatDB['Receptor']

tensor_database = pd.concat([spaCI, CellChatDB, iTALK], axis=0)
tensor_database = tensor_database.drop_duplicates()
tensor_database.index = list(range(len(tensor_database)))
tensor_database


# In[90]:


len(set(tensor_database['Ligand'])),len(set(tensor_database['Receptor']))


# In[ ]:


def ff(vv):
    vv[np.isinf(vv)] = 10 ** 8
    vv[np.isnan(vv)] = 0.001
    vv[vv == 0] = 0.001
    return vv
def model3(r1, r2, x, p, lamda, iter_num, stop_num):
    g = tl.tensor(np.array(random.sample(range(1,x.shape[2]*r1*r2+1),x.shape[2]*r1*r2), dtype=np.float64).reshape(r1, r2, x.shape[2]))
    u = np.array(random.sample(range(1,x.shape[1]*r1+1),x.shape[1]*r1), dtype=np.float64).reshape(x.shape[1], r1)
    v = np.array(random.sample(range(1,x.shape[0]*r2+1),x.shape[0]*r2), dtype=np.float64).reshape(x.shape[0], r2)
    chazhi1 = []
    chazhi2 = []
    for iter in range(iter_num):
        print("第 " + str(iter) + " 次迭代开始！！！！！！！！！！！")
        t = 0
        f = 0
        R = 0
        for i in range(x.shape[2]):
            # print(i)
            xt_v_gt = np.dot(np.dot(x[:, :, i].T, v), g[:, :, i].T)
            u_g_vt_c_gt = np.dot(np.dot(np.dot(np.dot(u, g[:, :, i]), v.T), v), g[:, :, i].T)

            u = np.multiply(u, xt_v_gt) / u_g_vt_c_gt
            u = ff(u)

            x_u_g = np.dot(np.dot(x[:, :, i], u), g[:, :, i])
            v_gt_ut_u_g = np.dot(np.dot(np.dot(np.dot(v, g[:, :, i].T), u.T), u), g[:, :, i])

            v = np.multiply(v, x_u_g) / v_gt_ut_u_g
            v = ff(v)

            ut_u_g_vt_v = np.dot(np.dot(np.dot(np.dot(u.T, u), g[:, :, i]), v.T), v)

            pgi = 0
            for j in range(x.shape[2]):

                pgi = pgi + p[i, j] * g[:, :, i]

            lpgi = lamda * pgi
            # print(lpgi)

            value1 = ut_u_g_vt_v + lpgi

            ut_xt_v = np.dot(np.dot(u.T, x[:, :, i].T), v)

            pgj = 0
            for j in range(x.shape[2]):
                pgj = pgj + p[i, j] * g[:, :, j]
            lpgj = lamda * pgj

            value = (g[:, :, i] * (ut_xt_v + lpgj)) / value1

            # value = ff(value)

            g[:, :, i] = value  #默认构造数组为整型，赋值小数直接向下取整
            # print('eeeeeeeeeeeeee:',g[:,:,i])

            t = t + np.linalg.norm(x[:, :, i] - np.dot(np.dot(v, g[:, :, i].T), u.T), 'fro')
            gp_sum = 0
            for k in range(x.shape[2]):
                gp_sum = gp_sum + p[i, k] * np.linalg.norm(g[:, :, i] - g[:, :, k], 'fro')

            f = f + (lamda / 2) * gp_sum
        R = t + f

        chazhi1.append(R)

        if iter == 0:
            chazhi = chazhi1[0]
            chazhi2.append(chazhi)
        else:
            chazhi = abs(chazhi1[iter] - chazhi1[iter - 1])
            chazhi2.append(abs(chazhi1[iter] - chazhi1[iter - 1]))

        print('T,F,Objective Function Value and |O(k+1) - O(k)|: ', t, f, R, chazhi)
        if chazhi < stop_num:
            print("在第 " + str(iter) + " 次迭代结束！")
            print(chazhi)
            break
        if iter == iter_num - 1 and chazhi >= stop_num:
            print(chazhi)

    # plt.rcParams["figure.figsize"] = (15, 4)
    # plt.subplot(1, 2, 1)
    # plt.plot(range(iter+1), chazhi1, 'r', label='Training loss')
    # plt.title('The stopping criterion is |O(k+1) - O(k)|<'+str(stop_num))
    # plt.xlabel("Number of iterations")
    # plt.ylabel("Objective Function Value")
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(range(iter+1), chazhi2, 'r', label='Training loss')
    # plt.title('The stopping criterion is |O(k+1) - O(k)|<'+str(stop_num))
    # plt.xlabel("Number of iterations")
    # plt.ylabel("|Ok+1 - Ok|")
    #
    # plt.savefig("iter.png")

    x_new = tl.tensor(np.array(random.sample(range(1, x.shape[0] * x.shape[1] * x.shape[2] + 1),
                                             x.shape[0] * x.shape[1] * x.shape[2]),
                               dtype=np.float64).reshape(x.shape[0], x.shape[1], x.shape[2]))
    for i in range(x.shape[2]):
        x_new[:, :, i] = np.dot(np.dot(v, g[:, :, i].T), u.T)
    # x_new
    return iter, g, u, v, x_new


# In[121]:


weizhi = pd.DataFrame(np.random.randint(1, 100 , (160,2)),columns = ['x','y'],index = ['c-'+str(x) for x in range(160)])
weizhi


# In[125]:


spot_x = weizhi['x']
spot_y = weizhi['y']
#计算距离矩阵Fai_ij
xspot_x=np.column_stack((spot_x,np.zeros(160)))
yspot_y=np.column_stack((spot_y,np.zeros(160)))
xx = cdist(xspot_x,xspot_x)
yy = cdist(yspot_y,yspot_y)

def function_dis(x,y):
    result = x**2 + y**2
    return result
start =time.time()
#为了让不连续的矩阵也能用if...else..语句
function_vectordis = np.vectorize(function_dis)
Fai_ij = function_vectordis(xx,yy)
#替换None为0
Fai_ij[Fai_ij == None] = 0
#替换NaN为0
Fai_ij = np.nan_to_num(Fai_ij, nan=0)
end = time.time()
print('Running time: %s Seconds'%(end-start))
Fai_ij


# In[126]:


# d = np.random.randint(1, 100, (160,160))
d = Fai_ij
for di in range(len(d)):
    d[di,di] = 1


# In[128]:


#计算多个坐标点的中心点（质心）
#https://blog.csdn.net/FlizhN/article/details/108318268
#角度弧度转化的函数
# radians()方法：转换角度为弧度的
# degrees()方法：从弧度转换到角度
from math import cos, sin, atan2, sqrt, pi ,radians, degrees
def center_geolocation(geolocations):
    x = 0
    y = 0
    z = 0
    lenth = len(geolocations)
    for lon, lat in geolocations:
        lon = radians(float(lon))
        lat = radians(float(lat))
        x += cos(lat) * cos(lon)
        y += cos(lat) * sin(lon)
        z += sin(lat)

    x = float(x / lenth)
    y = float(y / lenth)
    z = float(z / lenth)

    rx = degrees(atan2(y, x))
    ry = degrees(atan2(z, sqrt(x * x + y * y)))

    # return (degrees(atan2(y, x)), degrees(atan2(z, sqrt(x * x + y * y))))
    return rx,ry


# In[129]:


def my_point(a,b):
    return [a,b]
weizhi['zuobiao'] = weizhi.apply(lambda row:my_point(row['x'],row['x']), axis=1)
weizhi


# In[131]:


data = np.zeros((8,3))
df = pd.DataFrame(data,columns=['x','y','cluster'])
for i in range(8):
    locations = list(weizhi.iloc[20*i:20*(i+1),:]['zuobiao'])
    df.iloc[i,0] = center_geolocation(locations)[0]
    df.iloc[i,1] = center_geolocation(locations)[1]
    df.iloc[i,2] = 'tyep-'+str(i)
df


# In[134]:


#计算距离矩阵Fai_ij
xspot_x=np.column_stack((df.x,np.zeros(len(df.x))))
yspot_y=np.column_stack((df.y,np.zeros(len(df.y))))
xx = cdist(xspot_x,xspot_x)
yy = cdist(yspot_y,yspot_y)

def function_dis(x,y):
    result = x**2 + y**2
    return result
start =time.time()
#为了让不连续的矩阵也能用if...else..语句
function_vectordis = np.vectorize(function_dis)
_Fai_ij = function_vectordis(xx,yy)
#替换None为0
_Fai_ij[_Fai_ij == None] = 0
#替换NaN为0
_Fai_ij = np.nan_to_num(_Fai_ij, nan=0)
end = time.time()
print('Running time: %s Seconds'%(end-start))
_Fai_ij
for i in range(len(_Fai_ij)):
    _Fai_ij[i,i] = 1
_Fai_ij


# In[135]:


def s_data(type, num, type_n, num_spot, zubie):
    # type:
    # 1-强相关-交互；rho取[0.5,1]
    # 2-强相关-非交互；rho取[0,0.2]
    # 3-相似-交互；d取[0,1.5]
    # 4-相似-非交互;d取[5,10]
    # num:生成多少个值(LR对个数)
    # succ：第几次成功
    # 正态分布就生成N（0，1）的值
    # type_n:细胞的类别
    # zubie:几组数据
    vv = list(tensor_database['Ligand'][0:num])+list(tensor_database['Receptor'][0:num])+list(tensor_database['Receptor'][random.sample(list(range(num+1,len(tensor_database),1)), num)])+['cell_type']
    hh =['c-'+str(x) for x in range(num_spot)]
    num_type_spot = int(num_spot/type_n)
    p_ku = list(np.linspace(0.90,1,type_n))
    succ_ku = [3]
    # type_str = list(range(type_n))
    rho_i = list(np.linspace(0.5,1,zubie))
    rho_no_i = list(np.linspace(0,0.2,zubie))
    f_i = list(np.linspace(0,1.5,zubie))
    f_no_i = list(np.linspace(5,10,zubie))
    lr_list = []
    for zubiei in range(zubie):
        df_lr = pd.DataFrame(np.random.random((num_spot,3*num+1)), columns=vv, index =hh)
        for typei in range(type_n):
            p = p_ku[typei]
            # 二项分布考虑掷色子18次，成功次数符合二项分布
            N = num
            # 负二项分布考虑掷出成功n次，失败次数符合二项分布
            n = succ_ku[0]
            # 负二项分布NB(n,p)
            P_NB = stats.nbinom(n, p)
            # 成功3次，失败次数 k 的概率分布
            k = np.arange(N)
            gl = P_NB.pmf(k)
            gl = np.around(gl, decimals=3)#保留3位小数
            r = np.random.normal(0, 0.1, num)  #正态分布
            rho = random.sample(rho_i, 1)[0]
            rhon = random.sample(rho_no_i, 1)[0]
            f = random.sample(f_i, 1)[0]
            fn = random.sample(f_no_i, 1)[0]
            for i in range(num_type_spot):
                if type == 1:
                    for j in range(num_type_spot):
                        # print(i,j)
                        gr = np.array(list(map(lambda x: max(x, 0), rho * gl /d[num_type_spot*typei+i,num_type_spot*typei+j] + r)))
                        grn = np.array(list(map(lambda x: max(x, 0), rhon * gl /d[num_type_spot*typei+i,num_type_spot*typei+j] + r)))
                        gr = np.around(gr, decimals=3)#保留3位小数
                        grn = np.around(grn, decimals=3)
                        df_lr.iloc[typei*num_type_spot+i,0:num] = gl
                        df_lr.iloc[typei*num_type_spot+i,num:2*num] = gr
                        df_lr.iloc[typei*num_type_spot+i,2*num:3*num] = grn
                        df_lr.iloc[typei*num_type_spot+i,3*num] = 'type-'+str(typei)
                elif type == 2:
                    for j in range(num_type_spot):
                        gr = np.array(list(map(lambda x: max(x, 0), gl + f*d[num_type_spot*typei+i,num_type_spot*typei+j] + r)))
                        grn = np.array(list(map(lambda x: max(x, 0), gl + fn*d[num_type_spot*typei+i,num_type_spot*typei+j] + r)))
                        gr = np.around(gr, decimals=3)#保留3位小数
                        grn = np.around(grn, decimals=3)
                        df_lr.iloc[typei*num_type_spot+i,0:num] = gl
                        df_lr.iloc[typei*num_type_spot+i,num:2*num] = gr
                        df_lr.iloc[typei*num_type_spot+i,2*num:3*num] = grn
                        df_lr.iloc[typei*num_type_spot+i,3*num] = 'type-'+str(typei)
                else:
                    print('请输入正确的类型标识，type只能取[1，2]中的一个！')
        lr_list.append(df_lr)
    return lr_list


# In[98]:


num = 1000
df1 = pd.DataFrame()
df1['L'] = list(tensor_database['Ligand'][0:num])+list(tensor_database['Ligand'][0:num])
df1['R'] = list(tensor_database['Receptor'][0:num])+list(tensor_database['Receptor'][random.sample(list(range(num+1,len(tensor_database),1)), num)])
df1['ground_truth'] = list([1]*num)+list([0]*num)


# In[136]:


# s_data(type, num, type_n, num_spot, zubie):
    # type:
    # 1-强相关-交互；rho取[0.5,1]
    # 2-强相关-非交互；rho取[0,0.2]
    # 3-相似-交互；d取[0,1.5]
    # 4-相似-非交互;d取[5,10]
    # num:生成多少个值(LR对个数)
    # succ：第几次成功
    # 正态分布就生成N（0，1）的值
    # type_n:细胞的类别
    # zubie:几组数据
LR_list1 = s_data(1, 1000, 8, 160,10)
LR_list2 = s_data(2, 1000, 8, 160,10)


# In[108]:


LR_list = s_data(1, 8, 2, 20,10)
LR_list[1]


# In[137]:


LR_list1[1]


# In[147]:


LR_list2[1]


# In[142]:


for j in [1,2]:
    if j == 1:
        value_use = LR_list1
    else:
        value_use = LR_list2
    for i in range(len(value_use)):
        x = pd.DataFrame(value_use[i])
        if not os.path.exists('D://zhuomian//sim//data//data_'+str(j)):
            os.makedirs('D://zhuomian//sim//data//data_'+str(j))
        x.to_csv('D://zhuomian//sim//data//data_'+str(j)+'//4_3_data_test'+str(i)+'.txt',sep='\t',index=False)
        pd.DataFrame(df1).to_csv('D://zhuomian//sim//data//data_'+str(j)+'//ground_t'+str(i)+'.txt',sep='\t',index=False)


# In[156]:


result_zong = []
for k in [1,2]:
    result = []
    for j in range(10):
        Pre_value = []
        ground_t = pd.read_csv('D://zhuomian//sim//data//data_'+str(k)+'//ground_t'+str(j)+'.txt',sep='\t')
        r = pd.read_csv('D://zhuomian//sim//data//data_'+str(k)+'//4_3_data_test'+str(j)+'.txt',sep='\t')
        ground_t['LR'] = ground_t['L']+'-'+ground_t['R']

        x=r
        cell_type_mean = x.groupby(x['cell_type']).mean()
        n1 = cell_type_mean.iloc[:, 0:int(len(x.columns)/3)]
        n2 = cell_type_mean.iloc[:, int(len(x.columns)/3):2 * int(len(x.columns)/3)]
        n3 = cell_type_mean.iloc[:, int(len(x.columns)/3)*2:3 * int(len(x.columns)/3)]
        p = []
        for i in range(len(cell_type_mean.iloc[:, 0:int(2*len(x.columns)/3)].columns)):
            c_c = np.sqrt(np.matmul(np.array(pd.concat([n1,n1],axis=1).iloc[:, i]).reshape(-1, 1),
                                    np.array([list(pd.concat([n2,n3],axis=1).iloc[:, i])])))
            c_c = np.around(c_c, decimals=2)
            p.append(c_c)

        #距离矩阵
        LR_Score_c = _Fai_ij
        p_tensor = model3(3,5,np.array(p),np.array(LR_Score_c),10**4,50,0.1)[4]
        print(p_tensor)

        lll = []
        postive = []
        list1 = list([1] * int(len(x.columns)/3))+list([0] * int(len(x.columns)/3))
        test = tl.tensor(p_tensor, dtype=np.float64).reshape(len(p_tensor), len(p_tensor[0]), len(p_tensor[0]))

        list2 = []
        list3 = []
        list4 = []
        if k ==1:
            yuzhi = 0.4
        else:
            yuzhi = 0.4
        for ii in range(len(p_tensor[0])):
            for jj in range(len(p_tensor[0])):
                pt = p_tensor[:,ii,jj]
                list4 = list4 + list(np.array(pt).argsort()[-int(yuzhi*len(pt)):])

        mm = [0]*int(2*len(x.columns)/3)

        for rr in list(set(list4)):
            mm[rr] = 1
        postive=mm
        result.append(f1_score(postive, list1, average='macro'))
    result_zong.append(result)


