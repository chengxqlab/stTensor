#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def ff(vv):
    vv[np.isinf(vv)] = 10 ** 8
    vv[np.isnan(vv)] = 0.001
    vv[vv == 0] = 0.001
    return vv


def model3(r1, r2, x, p, lamda, iter_num, stop_num):
    # g = tl.tensor(np.array(random.sample(range(1, x.shape[2] * r1 * r2 + 1), x.shape[2] * r1 * r2),dtype=np.float64).reshape(r1,r2,x.shape[2]))
    # u = np.array(random.sample(range(1, x.shape[1] * r1 + 1), x.shape[1] * r1), dtype=np.float64).reshape(x.shape[1],
    #                                                                                                       r1)
    # v = np.array(random.sample(range(1, x.shape[0] * r2 + 1), x.shape[0] * r2), dtype=np.float64).reshape(x.shape[0],
    #                                                                                                       r2)

    g = tl.tensor(np.random.random((r1,r2,x.shape[2])))
    u = np.random.random(size=(x.shape[1],r1))
    v = np.random.random(size=(x.shape[0],r2))

    chazhi1 = []
    chazhi2 = []
    chazhi_T = []
    chazhi_F = []
    chazhi_T1 = []
    chazhi_F1 = []
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
        chazhi_T.append(t)
        chazhi_F.append(f)

        if iter == 0:
            chazhi = chazhi1[0]
            chazhi2.append(chazhi)
            chazhit = chazhi_T[0]
            chazhif = chazhi_F[0]
            chazhi_T1.append(chazhi_T[0])
            chazhi_F1.append(chazhi_F[0])
        else:
            chazhi = abs(chazhi1[iter] - chazhi1[iter - 1])
            chazhi2.append(abs(chazhi1[iter] - chazhi1[iter - 1]))
            chazhit = abs(chazhi_T[iter] - chazhi_T[iter - 1])
            chazhif = abs(chazhi_F[iter] - chazhi_F[iter - 1])
            chazhi_T1.append(abs(chazhi_T[iter] - chazhi_T[iter - 1]))
            chazhi_F1.append(abs(chazhi_F[iter] - chazhi_F[iter - 1]))

        # print('T,F,Objective Function Value and |O(k+1) - O(k)|: ', t, f, R, chazhi)
        print('dT,dF,Objective Function Value and |O(k+1) - O(k)|: ', chazhit, chazhif, R, chazhi)

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
    return iter, g, u, v, x_new, chazhi_T1, chazhi_F1

