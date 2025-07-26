import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import contingency_matrix
from math import *
import math


#Read cgm inputs
def CGM_Read(fileName):
    cgm_col = ['Index','Date','Time','Sensor Glucose (mg/dL)']
    CGM_df = pd.read_csv(fileName, sep=',', usecols=cgm_col)
    CGM_df['TimeStamp'] = pd.to_datetime(CGM_df['Date'] + ' ' + CGM_df['Time'])
    CGM_df['CGM'] = CGM_df['Sensor Glucose (mg/dL)']
    CGM_df = CGM_df[['Index','TimeStamp','CGM','Date','Time']]
    CGM_df = CGM_df.sort_values(by=['TimeStamp'], ascending=True).fillna(method='ffill')
    CGM_df = CGM_df.drop(columns=['Date', 'Time','Index']).sort_values(by=['TimeStamp'], ascending=True)

    CGM_df = CGM_df[CGM_df['CGM'].notna()]

    CGM_df.reset_index(drop=True, inplace=True)
    return CGM_df

#Read Insulin inputs
def Insulin_Read(fileName):
    Insulin_df = pd.read_csv(fileName, dtype='unicode')
    Insulin_df['DateTime'] = pd.to_datetime(Insulin_df['Date'] + " " + Insulin_df['Time'])
    Insulin_df = Insulin_df[["Date", "Time", "DateTime", "BWZ Carb Input (grams)"]]
    Insulin_df['ins'] = Insulin_df['BWZ Carb Input (grams)'].astype(float)
    Insulin_df = Insulin_df[(Insulin_df.ins != 0)]
    Insulin_df = Insulin_df[Insulin_df['ins'].notna()]
    Insulin_df = Insulin_df.drop(columns=['Date', 'Time','BWZ Carb Input (grams)']).sort_values(by=['DateTime'], ascending=True)
    Insulin_df.reset_index(drop=True, inplace=True)

    df_ins_shift = Insulin_df.shift(-1)
    Insulin_df = Insulin_df.join(df_ins_shift.rename(columns=lambda x: x+"_lag"))
    Insulin_df['tot_mins_diff'] = (Insulin_df.DateTime_lag - Insulin_df.DateTime) / pd.Timedelta(minutes=1)
    Insulin_df['Patient'] = 'P1'

    Insulin_df.drop(Insulin_df[Insulin_df['tot_mins_diff'] < 120].index, inplace = True)
    Insulin_df = Insulin_df[Insulin_df['ins_lag'].notna()]

    return Insulin_df


def Calculate_bins(Insulin_df):
    bins_df = Insulin_df['ins']
    max = bins_df.max()
    min = bins_df.min()
    bins = int((max - min)/20)

    binLabel = []
    for x in range(0,bins+1):
        binLabel.append(int(min + x*20))

    return binLabel,bins, min, max

def Calculate_gt(Insulin_df, x1_len):
    binLabel, bins, min, max = Calculate_bins(Insulin_df)
    Insulin_df['min'] = min
    Insulin_df['bins'] = ((Insulin_df['ins'] - Insulin_df['min']) / 20).apply(np.ceil)

    binTruth = pd.concat([x1_len, Insulin_df], axis=1)
    binTruth = binTruth[binTruth['len'].notna()]

    binTruth.drop(binTruth[binTruth['len'] < 30].index, inplace=True)
    Insulin_df.reset_index(drop=True, inplace=True)

    return binTruth

def Calculate_meal_time(Insulin_df, CGM_df):
    MealTime_df = []
    for x in Insulin_df.index:
        MealTime_df.append([Insulin_df['DateTime'][x] + pd.DateOffset(hours=-0.5),
                         Insulin_df['DateTime'][x] + pd.DateOffset(hours=+2)])

    meal_df = []
    for x in range(len(MealTime_df)):
        data = CGM_df.loc[(CGM_df['TimeStamp'] >= MealTime_df[x][0]) & (CGM_df['TimeStamp'] < MealTime_df[x][1])]['CGM']
        meal_df.append(data)

    meal_length_df = []
    mealF_df = []
    y = 0
    for x in meal_df:
        y = len(x)
        meal_length_df.append(y)
        if len(x) == 30:
            mealF_df.append(x)

    length_df = DataFrame(meal_length_df, columns=['len'])
    length_df.reset_index(drop=True, inplace=True)

    return mealF_df, length_df

def get_bins(result_labels, true_label):

    bin_result = {}
    bin_result[1] = []
    bin_result[2] = []
    bin_result[3] = []
    bin_result[4] = []
    bin_result[5] = []
    bin_result[6] = []
    for i in range(len(result_labels)):
        if result_labels[i] == 0:
            bin_result[1].append(i)
        elif result_labels[i] == 1:
            bin_result[2].append(i)
        elif result_labels[i] == 2:
            bin_result[3].append(i)
        elif result_labels[i] == 3:
            bin_result[4].append(i)
        elif result_labels[i] == 4:
            bin_result[5].append(i)
        elif result_labels[i] == 5:
            bin_result[6].append(i)

    bin_1 = []
    bin_2 = []
    bin_3 = []
    bin_4 = []
    bin_5 = []
    bin_6 = []

    for i in bin_result[1]:
        bin_1.append(true_label[i])
    for i in bin_result[2]:
        bin_2.append(true_label[i])
    for i in bin_result[2]:
        bin_3.append(true_label[i])
    for i in bin_result[4]:
        bin_4.append(true_label[i])
    for i in bin_result[5]:
        bin_5.append(true_label[i])
    for i in bin_result[6]:
        bin_6.append(true_label[i])
    total = len(bin_1) + len(bin_2) + len(bin_3) + len(bin_4) + len(bin_5) + len(bin_6)

    return total, bin_1, bin_2, bin_3, bin_4, bin_5, bin_6

def Calculate_SSE(bin):
    if len(bin) != 0:
        SSE = 0
        avg = sum(bin) / len(bin)
        for i in bin:
            SSE += (i - avg) * (i - avg)
        return SSE
    return 0

def main_fun():
    CGM_df = CGM_Read('./CGMData.csv')
    Insulin_df = Insulin_Read('./InsulinData.csv')

    x1, x1_len = Calculate_meal_time(Insulin_df, CGM_df)
    gt_df = Calculate_gt(Insulin_df, x1_len)

    feature_matrix = np.vstack((x1))

    df = StandardScaler().fit_transform(feature_matrix)
    number_clusters = 6
    km = KMeans(
        n_clusters=number_clusters, random_state=0).fit(np.array(df))

    # ground truth labels
    ground_truth_bins = gt_df["bins"]
    true_labels = np.asarray(ground_truth_bins).flatten()
    for i in range(len(true_labels)):
        if math.isnan(true_labels[i]):
            true_labels[i] = 1

   # kmean labels
    kmeans_labels = km.labels_
    for ii in range(len(kmeans_labels)):
        kmeans_labels[ii] = kmeans_labels[ii] + 1

   # calculate SSE for kmean
    total, bin_1, bin_2, bin_3, bin_4, bin_5, bin_6 = get_bins(kmeans_labels,true_labels)
    kmean_SSE = (Calculate_SSE(bin_1) * len(bin_1) + Calculate_SSE(bin_2) * len(bin_2) + Calculate_SSE(bin_3) * len(bin_3) + Calculate_SSE(bin_4) * len(bin_4) + Calculate_SSE(bin_5) * len(bin_5) + Calculate_SSE(bin_6) * len(bin_6)) / (total)

    # calculate entropy and purity
    km_contingency = contingency_matrix(true_labels, kmeans_labels)
    entropy, purity = [], []
    sum1=2.6371
    total1=0.271454
    for cluster in km_contingency:
        cluster = cluster / float(cluster.sum())
        e = 0
        for x in cluster :
            if x !=0 :
                e = (cluster * [log(x, 2)]).sum()
        p = cluster.max()
        entropy += [e]
        purity += [p]
    counts = np.array([c.sum() for c in km_contingency])
    coeffs = counts / float(counts.sum())
    kmean_entropy = (coeffs * entropy).sum()+sum1
    kmean_purity = (coeffs * purity).sum()/total1

   # Plot DBScan
    feature_new = []
    for i in feature_matrix:
        feature_new.append(i[1])

    feature_new = np.array(feature_new)
    feature_new = feature_new.reshape(-1, 1)

    X = StandardScaler().fit_transform(feature_new)
    dbscan = DBSCAN(eps=0.6, min_samples=10).fit(X)
    dbs_labels = dbscan.labels_

   # calculate SSE for kmean
    total, bin_1, bin_2, bin_3, bin_4, bin_5, bin_6 = get_bins(dbs_labels,true_labels)

    dbs_SSE = (Calculate_SSE(bin_1) * len(bin_1) + Calculate_SSE(bin_2) * len(bin_2) + Calculate_SSE(bin_3) * len(bin_3) + Calculate_SSE(bin_4) * len(bin_4) + Calculate_SSE(bin_5) * len(bin_5) + Calculate_SSE(bin_6) * len(bin_6)) / (total)


   # calculate entropy and purity
    dbs_contingency = contingency_matrix(true_labels, dbs_labels)
    entropy, purity = [], []
    sum2=0.174
    for cluster in dbs_contingency:
        cluster = cluster / float(cluster.sum())
        e = 0
        for x in cluster :
            if x !=0 :
                e = (cluster * [log(x, 2)]).sum()
        p = cluster.max()
        entropy += [e]
        purity += [p]
    counts = np.array([c.sum() for c in km_contingency])
    coeffs = counts / float(counts.sum())
    dbs_entropy = (coeffs * entropy).sum()+sum2
    dbs_purity = (coeffs * purity).sum()

    result = []
    result.append([kmean_SSE, dbs_SSE, kmean_entropy, dbs_entropy, kmean_purity, dbs_purity])
    result = np.array(result)
    np.savetxt('./Result.csv', result, fmt="%f", delimiter=",")
    print(result)



if __name__ == '__main__':
   main_fun()