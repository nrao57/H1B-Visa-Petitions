import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

def get_data(path_to_data, chunk='all'):
    '''
    This function extracts the data from csv file to pandas dataframe
    '''
    if chunk == 'all':
        dwdata = pd.read_csv(path_to_data)
    else:
        dwdata = pd.read_csv(path_to_data, nrows = chunk)
		
    return dwdata

def format_clean(data):
    '''
    This function does some basic transformation on data
    We want to remove the 'Withdrawn' from the target labeled dataset 
    and merge 'Certified-Withdrawn' with 'Certified' in 
    order to make the y_labels/targets binary 
    '''
    data2 = data.drop('Unnamed: 0', axis = 1)
    df1 = data2[data2['CASE_STATUS']!='WITHDRAWN']
    df2 =df1[df1['CASE_STATUS']!='CERTIFIED-WITHDRAWN']
    
    #ONE-HOT Encoding of 
    df3 = pd.get_dummies(df2.drop('CASE_STATUS', axis = 1))
    df4 = pd.concat([df3, df2['CASE_STATUS']], axis=1)
    df5 = df4.dropna()
    
    return df5


def visualize(data):
    f1, ax1 = plt.subplots(1)
    lons,lats = data['lon'].values,data['lat'].values
    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
    # are the lat/lon values of the lower left and upper right corners
    # of the map.
    # lat_ts is the latitude of true scale.
    # resolution = 'c' means use crude resolution coastlines.
    m = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=55,
            llcrnrlon=-135,urcrnrlon=-60,lat_ts=20,resolution='l')
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    #m.drawmapboundary(fill_color='aqua')
    x,y = m(lons,lats)
    m.scatter(x,y,marker='o')
    plt.title("Mercator Projection of H-1B Visa Destinations")
    
    
    
    f3, ax3 = plt.subplots(1)
    data.hist('PREVAILING_WAGE', bins = 1000, ax=ax3)
    ax3.set_xlim([0,150000])
    ax3.set_xlabel('Wage ($)')
    ax3.set_ylabel('Number of Applicants')

    plt.show()

    


