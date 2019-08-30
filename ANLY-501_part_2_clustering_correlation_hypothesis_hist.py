#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 11:07:51 2018

@author: yxy
"""
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from scipy import stats

from scipy.sparse import *
from scipy import *

from sklearn.decomposition import PCA

import seaborn as sns; sns.set()  # for plot styling
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import preprocessing
from sklearn import multiclass

import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import MNLogit
import seaborn as sns

################################################################################
def DBScanClustering(DFArray, eps, min_samples=10):
    #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    DBSCAN_fit = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(DFArray)
    core_samples_mask = np.zeros_like(DBSCAN_fit.labels_, dtype=bool)
    core_samples_mask[DBSCAN_fit.core_sample_indices_] = True
    labels = DBSCAN_fit.labels_
    #print(labels)
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  
    ## Plot the result
    ## Plot code from here:
    ## http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = DFArray[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=2)

        xy = DFArray[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    #plt.xlabel(DFArray[:,0])
    #plt.ylabel(DFArray[:,1])
    plt.show()
    plt.close()

    ss=metrics.silhouette_score(DFArray, labels, metric='euclidean')
    print("Silouette score of this DBscan is",ss)
    
def kmeansClustering(DFArray, DistanceMeasure="L2", k=3):
    DF=DFArray
    kmeansResults=KMeans(n_clusters=k,init='k-means++', verbose=1, algorithm="full")
    kmeansResults.fit(DF)
    #print("Centers: ", kmeansResults.cluster_centers_)  
    #print("Labels: ", kmeansResults.labels_)
    #print("Intertia (L2norm dist):", kmeansResults.inertia_)


def MakeBlobs_kmeans(X,k):
    ## Call kmeans on the blobs
    kmeansResultsB=KMeans(n_clusters=k, verbose=1, precompute_distances=True)
    kmeansResultsB.fit(X)
    y_kmeans = kmeansResultsB.predict(X)
    #print("Centers: ", kmeansResultsB.cluster_centers_)  
    #print("Labels: ", kmeansResultsB.labels_)
    #print("Intertia (L2norm dist):", kmeansResultsB.inertia_)
    ## VIS
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeansResultsB.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.title("Plot of Kmeans")
    plt.show()
    plt.close()
    #return X
    
    labels=kmeansResultsB.labels_
    ss=metrics.silhouette_score(X, labels, metric='euclidean')
    print("Silouette score of Kmeans is",ss)

def Makeanarray(x1,x2):
    a0 =[]
    for i in range(len(x1)):
        a0.append([x1[i],x2[i]])
    a0=np.asarray(a0)
    return a0

def MP_Categorize(df,x):
    for i in range(len(x)):
        if (x[i]=='New Moon'):
            df.ix[i,'MPC']=1
        elif (x[i]=='Waxing Crescent'):
            df.ix[i,'MPC']=2
        elif (x[i]=='First Quarter'):
            df.ix[i,'MPC']=3
        elif (x[i]=='Waxing Gibbous'):
            df.ix[i,'MPC']=4
        elif (x[i]=='Full Moon'):
            df.ix[i,'MPC']=5
        elif (x[i]=='Waning Gibbous'):
            df.ix[i,'MPC']=6
        elif (x[i]=='Wanning Crescent'):
            df.ix[i,'MPC']=7
        else:
            df.ix[i,'MPC']=8
    return df

def prepanalysis(x):
    labels=['low','medium','high']
    x=pd.qcut(x, 3, labels=labels)
    return x

def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x - n*m_y*m_x) 
    SS_xx = np.sum(x*x - n*m_x*m_x) 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 
  
def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show()

def ward(X):
    linkage_matrix = linkage(X, 'ward')
    figure = plt.figure(figsize=(7.5, 5))
    dendrogram(
            linkage_matrix,
            color_threshold=0,
    )
    plt.title('Hierarchical Clustering Dendrogram (Ward)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.tight_layout()
    plt.show()
    
    #ss=metrics.silhouette_score(X, labels, metric='euclidean')
    #print("Silouette score of this WARD is",ss)
    
def linearR(X,Y):    
    #X2 = sm.add_constant(X)
    est = sm.OLS(Y, X)
    est2 = est.fit()
    print(est2.summary())
    
def get_dummy(da):
    s=pd.get_dummies(da['Moon_Phase'])
    result = pd.concat([da,s], axis=1)
    result=result.drop(['Moon_Phase'],axis=1)
    return result    

################################################################################
def boxpl(db):
    fig = plt.figure(1, figsize=(9, 6))

    ax = fig.add_subplot(111)
    bp = ax.boxplot(db, patch_artist=True)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
    # change outline color
        box.set( color='#7570b3', linewidth=2)
    # change fill color
        box.set( facecolor = '#1b9e77' )

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    
    ax.set_ylim([-0.1,0.1])
    
def corr(x):
    # correlation table
    corr = plt.corr()
    ax = plt.axes()
    sns.heatmap(corr,
                annot=True,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    ax.set_title("Correlation Heat Map", size = 15)
    name1="Correlation"+str(x)
    plt.savefig(name1)
    plt.close
    
    # subplots of scatterplots
    sns.pairplot(x)
    name2="Scatterplots"+str(x)
    plt.savefig(name2)
    plt.close

def hist(df):
    plt.subplot(131)
    plt.hist(df["max"])
    plt.xlabel('Crime Count')
    plt.ylabel('Frequency')
    plt.title("Total Crime")
    plt.subplot(132)
    plt.hist(df["Pub_Sch_Pop"])
    plt.title("The number of public schools")
    plt.xlabel('Public School Count')
    plt.ylabel('Frequency')
    plt.subplot(133)
    plt.hist(df["Pri_Sch_Pop"])
    plt.title("The number of Private Schools")
    plt.xlabel('Private School Count')
    plt.ylabel('Frequency')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.4,
                        wspace=0.35)    
    plt.savefig("Histogram")
    plt.show()
    plt.close
    
###############################################################################################################
    
def main():
    # Data preperation
    df=pd.read_csv("crimedata_final_final.csv")
    #df0=pd.concat([df['Pri_Sch_Cnt'],df['Pub_Sch_Cnt'],df['Uni_Cnt']], axis=1)
    df['max'] = df['max'].map({'Violence that Involved Property': 0, 
                                 'Acts Causing Harm to Person': 1,'Controlled Substances':2,
                                'Fraud, Deception, or Corruption':3,'Crimes Leading/Intending to Death':4,
                               'Injuries Acts of a Sexual Nature':5 })
    df=get_dummy(df)    
    df=df.sample(frac=0.01)
    df=df.reset_index()
    #np.sum(df.iloc[:,4:11].values,axis=1)
    
    ## Kmeans & DBscan with crime rates and Sun Hour
    array1=Makeanarray(df['Violence that Involved Property'],np.mean(df.iloc[:,13:15].values,axis=1))
    
    ## Hierarchical Clustering with Crime Types and Public School Population
    array3=Makeanarray(df['max'],df['Pub_Sch_Pop'])

    
   ####################################################################
    
    ## Kmeans Violence that Involved Property vs. Average Temperature
    MakeBlobs_kmeans(array1,2)
    
    ## DBscan 
    DBScanClustering(array1,5)
    
    ## Hierarchical
    ward(array3)

   ######################################################################
    
    # Correlation Plots
    df2=df[['Acts Causing Harm to Person','Controlled Substances','Crimes Leading/Intending to Death','Fraud, Deception, or Corruption','Injuries Acts of a Sexual Nature','Violence that Involved Property','Moon_Illumination','MaxTemperature','MinTemperature','SunHour','adult_obesity','adult_smoking','Weekend','Month','Pri_Sch_Cnt','Pub_Sch_Cnt', 'Day']]
    Var_Corr = df2.corr()
    plt.subplots(figsize=(20,15))
    # plot the heatmap and annotation on it
    sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
    
    # Histogram
    hist(df)
    
   #####################################################################    

    ## T-test 
    ttest=stats.ttest_ind(array3[:,0],array3[:,1], equal_var = False)
    print("\nP-value is",ttest.pvalue)
    df=df.dropna()
    
    ## Logistic Regression
    x=pd.concat([df['Moon_Illumination'],df['MaxTemperature'],df['MinTemperature'],df['SunHour'],df['adult_obesity'],df['adult_smoking'],df['Weekend'],df['Month'],df['Pri_Sch_Cnt'],df['Pub_Sch_Cnt'],df['Day']],axis=1)
    y=df['max']
    model=MNLogit.from_formula("y ~ x", df).fit()
    print(model.summary())
    
    ## Linear Regression
    X = df[['Moon_Illumination','MaxTemperature','MinTemperature','SunHour']]
    Y= df['max']
    s1=linearR(X,Y)
    print(s1)

###############################################################################################################

if __name__ == "__main__":
    main()