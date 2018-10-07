import numpy as np

class Bdc():

    @staticmethod
    def __calculate_fc(set, label):
        rows = set.shape[0]
        csum = set.sum(axis=1)
        fc = dict()
        for i in range(rows):
            currentlabel = label[i]
            if not currentlabel in fc:
                fc[currentlabel] = 0
            fc[currentlabel] += csum[i]
        #print('fc succeed')
        #print(fc)
        return fc

    @staticmethod
    def __calculate_ftc(fc, set, label, K):
        cols = set.shape[1]
        rows = set.shape[0]
        ftc = np.matrix(np.zeros([K,cols]))
        counti = 0
        for i in fc:
            #print(i)
            for j in range(cols):
                for k in range(rows):
                    if label[k] == i:
                        ftc[counti,j] += set[k,j]
            counti += 1
        #print('ftc succeed')
        return ftc

    @staticmethod
    def __calculate_ptc(ftc, fcv, set, K):
        cols = set.shape[1]
        ptc = ftc.copy()
        for i in range(K):
            for j in range(cols):
                ptc[i,j] = ftc[i,j] / fcv[i]
        #print('ptc succeed')
        return ptc

    @staticmethod
    def __calculate_bdc(ptc, set, K):
        cols = set.shape[1]
        bdc = np.zeros(cols)
        #print(bdc)
        for i in range(cols):
            tmp = 0
            for j in range(K):
                tttmp = 0
                for k in range(K):
                    tttmp += ptc[k,i]
                if not tttmp == 0:
                    ttmp = ptc[j,i] / tttmp
                    if not ttmp == 0:
                        tmp += ttmp * np.log(ttmp)
            bdc[i] = 1 + tmp / 3
        print('bdc succeed')
        return bdc

    @staticmethod
    def cal_bdc(data_vec, subject_list, k):
        fc = Bdc.__calculate_fc(data_vec, subject_list)
        fcv = list(fc.values())
        ftc = Bdc.__calculate_ftc(fc, data_vec, subject_list, k)
        ptc = Bdc.__calculate_ptc(ftc, fcv, data_vec, k)
        bdc = Bdc.__calculate_bdc(ptc, data_vec, k)
        return bdc
