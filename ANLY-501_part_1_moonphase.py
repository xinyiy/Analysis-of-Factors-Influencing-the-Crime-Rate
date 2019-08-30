#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:00:15 2019

@author: yxy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 13:01:03 2018

@author: yxy
"""

import urllib
from urllib.request import urlopen
import pandas as pd
import requests

#API:603b70a62d7541a687131630182911
#https://developer.worldweatheronline.com/premium-api-explorer.aspx

def Moonphase(q):
    
    # There's limitation for data requested each time so it's better to seperate the annual data into monthly
    start=['2017-01-01','2017-02-01','2017-03-01','2017-04-01','2017-05-01','2017-06-01','2017-07-01','2017-08-01','2017-09-01','2017-10-01','2017-11-01','2017-12-01']
    end=['2017-01-31','2017-02-28','2017-03-31','2017-04-30','2017-05-31','2017-06-30','2017-07-31','2017-08-31','2017-09-30','2017-10-31','2017-11-30','2017-12-31']
    
    # Create several lists for further appends
    moonphase=[]
    date=[]
    moonilluminationlist=[]
    maxtemp=[]
    mintemp=[]
    sh=[]
    
    # Parameters need to be specified: date, enddate and zipcode(q)
    for i in range(len(start)):
        baseURL="http://api.worldweatheronline.com/premium/v1/past-weather.ashx"
        zipURL=baseURL + "?" + urllib.parse.urlencode({
                'format': "json",
                'key':'603b70a62d7541a687131630182911',
                'date':start[i],
                'enddate':end[i],
                'q':q
                })  
        # Use request package to get data
        response=requests.get(zipURL)
        # Make sure the type is json
        response=response.json()
        #print(response)
     
        # Only append the columns we need for further analysis
        try:
            data_all=response['data']['weather']  
            for j in data_all:
                day=j['date']
                weather=j['astronomy'][0]['moon_phase']
                moonillumination=j['astronomy'][0]['moon_illumination']
                maxtemp0=j['maxtempF']
                mintemp0=j['mintempF']
                sh0=j['sunHour']
                moonphase.append(weather)
                date.append(day)
                moonilluminationlist.append(moonillumination)
                maxtemp.append(maxtemp0)
                mintemp.append(mintemp0)
                sh.append(sh0)
        except:
                moonphase.append(None)
                date.append(None)
                moonilluminationlist.append(None)
            
    #print(moonphase)
    #print(date)
    #print(moonilluminationlist)

    df=pd.DataFrame()
    df['Date']=date
    df['Moon_Phase']=moonphase
    df['Moon_Illumination']=moonilluminationlist
    df['MaxTemperature']=maxtemp
    df['MinTemperature']=mintemp
    df['SunHour']=sh
    df['City_Name']=q
    # Independent csvs are assigned to each city
    name='Outfile_Moonphase_'+str(q)+'.csv'
    df.to_csv(name,index=False)

def main():
    #Retrieve the moon phase data of specified areas by zipcodes
    citynames=pd.read_csv('CityNames.txt')
    for i in citynames:
        Moonphase(i)
    
    # Merge moon phase data
    df_new=pd.DataFrame()
    for i in citynames:
        name='Outfile_Moonphase_'+i+'.csv'
        print(name)
        df=pd.read_csv(name)
        #df=df.drop(columns='')
        df_new=df_new.append(df,ignore_index=True)
    df_new.to_csv('Merged Moonphase Data_1.csv',index=False)
    #Moonphase('LA')
    
###############################################################################
if __name__ == "__main__":
    main()
