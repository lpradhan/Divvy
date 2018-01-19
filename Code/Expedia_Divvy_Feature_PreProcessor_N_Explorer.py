#Expedia Divvy Challenge...
#Author: Ligaj Pradhan

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import loadDataFrameRowFunctions as ld
from sklearn.model_selection import train_test_split

class DataPreProcessorAndExplorer:
    
    def __init__(self, parentDataFolder, dataSet_Divvy_Trips_CSV_FILE_1, dataSet_Divvy_Trips_CSV_FILE_2, dataSet_Divvy_Stations_CSV_FILE, useOnlyFirstFile):
        self.df_trips = pd.DataFrame()
        self.df_stations = pd.DataFrame()
        self.df_dist = pd.DataFrame()
        self.parentDataFolder = parentDataFolder
        self.dataSet_Divvy_Trips_1 = dataSet_Divvy_Trips_CSV_FILE_1
        self.dataSet_Divvy_Trips_2 = dataSet_Divvy_Trips_CSV_FILE_2
        self.dataSet_Divvy_Stations = dataSet_Divvy_Stations_CSV_FILE
        self.useOnlyFirstFile = useOnlyFirstFile 
        
    def loadDatasets(self):        
        self.df_trips = pd.read_csv(self.dataSet_Divvy_Trips_1)
        if (not self.useOnlyFirstFile):
            tmp = pd.read_csv(self.dataSet_Divvy_Trips_2)
            self.df_trips = pd.concat([self.df_trips,tmp])
        self.df_stations = pd.read_csv(self.dataSet_Divvy_Stations)
    
    def cleanDatasets(self):
         self.df_trips.dropna(inplace=True)
         self.df_trips.drop_duplicates(inplace=True)
    
    def info(self):
        self.df_stations.info()
        self.df_trips.info()
        self.df_dist.info()
    
    def show(self):
        print(self.df_trips.head())
        print(self.df_stations.head())
        print(self.df_dist.head())
        
    def getStations(self):
        return self.df_stations
        
    def getTrips(self):
        return self.df_trips
        
    def getDist(self):
        return self.df_dist
    
    # returns normalized col of a dataframe
    def normalize(self, df, col):
        result = df[col].copy()
        max_value = result.max()
        min_value = result.min()
        result = (result - min_value) / (max_value - min_value)
        return result
        
    def getInterStationDistancesInKm(self):
        self.df_stations['key'] = 1
        self.df_dist = pd.merge(self.df_stations,self.df_stations,on='key').drop('key',axis=1)
        tmp = self.df_dist.apply(ld.getDistanceInKmFromLatitudeAndLongitude, axis=1)
        self.df_dist['distance'] = tmp.values
        self.df_dist = self.df_dist[['id_x','id_y','distance']].copy()
        #Saving inter station distances in kilometers in a csv file
        file = self.parentDataFolder + '/Inter_Station_Dist.csv'
        if (os.path.isfile(file)):
            print('Inter Station Distance file already available...')
        else:
            self.df_dist.to_csv(file, header=['id_x', 'id_y', 'distance'])

    def addGroundTruthColumn(self):
        # GroundTruth : 'SHORT' or 'LONG'
        self.df_trips = pd.merge(self.df_trips, self.df_dist, how='left', left_on=['from_station_id','to_station_id'], right_on=['id_x','id_y'])
        self.df_trips['GroundTruth'] = np.where(self.df_trips['distance']<2, 'SHORT', 'LONG')

    # Extract useful and more effective time features from the dates field and create separate columns for them...
    def extractTimeFeaturesFromDate(self):
        tmp = pd.to_datetime(self.df_trips.start_time)
        # Get day_of_week
        # Analysing the date field, days of week are encoded as : Mon = 0, Tue = 1, Wed = 2, Thur = 3, Fri = 4, Sat = 5, Sun = 6
        t = tmp.dt.weekday
        self.df_trips['day_of_week'] = t
        # Get hour in 24 hr format..
        t = tmp.dt.hour
        self.df_trips['hour_of_day'] = t
    
    # Extract useful and more interprepable User features from the dates field and create separate columns for them...
    def extractUserFeatures(self):
        #'rider_age'
        # Age should be more useful than birthyear while training the prediction model...
        self.df_trips['rider_age'] = self.df_trips.apply(lambda row: 2018-row.birthyear, axis=1)
        #'gender_num'
        #Lets represent gender with 1 and 0 in an additional column called 'gender_num'
        self.df_trips['gender_num'] = np.where(self.df_trips['gender']=='Male', 1, 0)
        #'Customer'|'Dependent'|'Subscriber'
        # Lets represent usertypes as categorical features in columns Customers, Dependent and Subscriber
        self.df_trips['Customer'] = np.where(self.df_trips['usertype']=='Customer', 1, 0)
        self.df_trips['Dependent'] = np.where(self.df_trips['usertype']=='Dependent', 1, 0)
        self.df_trips['Subscriber'] = np.where(self.df_trips['usertype']=='Subscriber', 1, 0)
     
    # Extract useful and more effective Station based features from the dates field and create separate columns for them...
    def extractStationFeatures(self):
        # Longitude and Latitude values are very close as we are considering places within a single city. 
        # Hence we normalize them between 0 and 1 to identify/represent their proximity or distance based on their geographcal location.
        normalized_col = self.normalize(self.df_stations, 'latitude')
        self.df_stations['lat_n'] = normalized_col
        normalized_col = self.normalize(self.df_stations, 'longitude')
        self.df_stations['long_n'] = normalized_col
    
    def getShortToLongTripCountRatioFor(self, df, col, max_threshold):
        tmp = df.groupby([col,'GroundTruth']).count()
        tmp = tmp.unstack()['gender']  #we just pick one column, can pick any...
        tmp = tmp.reset_index() #remove the multi level indexes and make them columns
        tmp.fillna(value=0.5, inplace=True) #To avoid divide by 0, we put 0.5 if the value is 0
        t = tmp.apply(ld.getShortByLongRatio, axis=1)
        tmp['ratioSL'] = t
        #We cap the highest ratioSL to max_threshold i.e. ratios more than 'max_threshold' will be made 10
        tmp['ratioSL'] = np.where(tmp['ratioSL']>max_threshold, max_threshold, tmp['ratioSL'])
        return tmp
    
    def plotVsratioSL(self, f, threshold):
        #ratioSL>threshole will be capped at threshold
        tmp = self.getShortToLongTripCountRatioFor(self.df_trips, f, threshold)
        plt.figure()
        sns.lmplot(x=f, y='ratioSL', data=tmp, size=4)
        ax = plt.gca()
        ax.set_title('ratioSL vs ' + f)
        plt.figure()
        sns.lmplot(x=f, y='SHORT', data=tmp, size=4)
        ax = plt.gca()
        ax.set_title('SHORT trip counts vs ' + f)
        plt.figure()
        sns.lmplot(x=f, y='LONG', data=tmp, size=4)
        ax = plt.gca()
        ax.set_title('LONG trip counts vs ' + f)
        
    def finalizeDataset(self):
        print('Finalizing...')
        
        # Lets add fromStation's longitude and latitude features along with each trip
        self.df_trips = pd.merge(self.df_trips, self.df_stations, how='left', left_on='from_station_id', right_on='id')
        
        #Divide into train(80%) validation(10%) and test(10%) datasets...
        train, test = train_test_split(self.df_trips, test_size=0.2, shuffle=True, random_state=12345)
        validation, test = train_test_split(test, test_size=0.5, shuffle=True, random_state=12345)
        
        # Compute LONG and SHORT trip counts and 'ratioSL' features for each fromStation based on only Training DataSet and append it as an additional feature of the fromStation
        f = 'from_station_id'
        tmp = self.getShortToLongTripCountRatioFor(train, f, 20); #max rationSL threshold set to 5
        # Normalize LONG and SHORT counts for stations...
        normalized_col = self.normalize(tmp, 'LONG')
        tmp['LONG'] = normalized_col
        normalized_col = self.normalize(tmp, 'SHORT')
        tmp['SHORT'] = normalized_col
        tmp = tmp.rename(columns={'LONG': 'LONG_Trips_FromStation', 'SHORT': 'SHORT_Trips_FromStation', 'ratioSL':'ratioSL_FromStation'})
        
        #Save this tmp along with normalized longitude and latitude as we will need it later to test new trips using the saved trained classifier model
        file = self.parentDataFolder + '/Station_Processed_Info.csv'
        df_station_data = pd.merge(self.df_stations, tmp, how='left', left_on='id', right_on='from_station_id')
        df_station_data.fillna(0, inplace=True)
        df_station_data = df_station_data.loc[:,['from_station_id','lat_n','long_n','LONG_Trips_FromStation','SHORT_Trips_FromStation','ratioSL_FromStation']]
        if (os.path.isfile(file)):
            print('Station_Processed_Info file already available...')
        else:
            df_station_data.to_csv(file, header=True)
        
        #Add LONG_Trips_FromStation SHORT_Trips_FromStation and ratioSL_FromStation to train validation and test datasets...
        train = pd.merge(train, tmp, on='from_station_id', how='left')
        validation = pd.merge(validation, tmp, on='from_station_id', how='left')
        test = pd.merge(test, tmp, on='from_station_id', how='left')
        # There can be some stations that didn't occur in the train data, they will introduce NaNs in the test and validation dataset after merging.
        # We replace such nans with 0
        validation.fillna(0, inplace=True)
        test.fillna(0, inplace=True)
        
        # Normalize GroundTruth as SHORT->1 and LONG->0
        train['GroundTruth'] = np.where(train['GroundTruth']=='SHORT', 1, 0)
        validation['GroundTruth'] = np.where(validation['GroundTruth']=='SHORT', 1, 0)
        test['GroundTruth'] = np.where(test['GroundTruth']=='SHORT', 1, 0)
        
        # Rearranding columns.....as | TripID | Station Features | User Features | Time Features | GroundTruth |
        train = train[['from_station_id', 'lat_n', 'long_n', 'LONG_Trips_FromStation', 'SHORT_Trips_FromStation', 'ratioSL_FromStation', 'rider_age', 'gender_num', 'Customer', 'Dependent', 'Subscriber', 'day_of_week', 'hour_of_day', 'GroundTruth']]
        validation = validation[['from_station_id', 'lat_n', 'long_n', 'LONG_Trips_FromStation', 'SHORT_Trips_FromStation', 'ratioSL_FromStation', 'rider_age', 'gender_num', 'Customer', 'Dependent', 'Subscriber', 'day_of_week', 'hour_of_day', 'GroundTruth']]
        test = test[['from_station_id', 'lat_n', 'long_n', 'LONG_Trips_FromStation', 'SHORT_Trips_FromStation', 'ratioSL_FromStation', 'rider_age', 'gender_num', 'Customer', 'Dependent', 'Subscriber', 'day_of_week', 'hour_of_day', 'GroundTruth']]
        
        # Save Train, Validation and Test datasets into csv files
        # Save Training dataset...
        print('Saving train, validation and test datasets...')
        file = self.parentDataFolder + '/train.csv'
        if (os.path.isfile(file)):
            print('Training data file already available...')
        else:
            train.to_csv(file, header=True)
        #Save Validation dataset...
        file = self.parentDataFolder + '/validation.csv'
        if (os.path.isfile(file)):
            print('Validation data file already available...')
        else:
            validation.to_csv(file, header=True)
            #Save Training dataset...
        file = self.parentDataFolder + '/test.csv'
        if (os.path.isfile(file)):
            print('Testing data file already available...')
        else:
            test.to_csv(file, header=True)

    def exploreFeatures(self):
        plt.figure()
        self.df_trips['day_of_week'].plot.hist(bins=np.arange(8)-0.5, ec='black',xticks=np.arange(7), title='Trip count vs day_of_week')
        plt.figure()
        self.df_trips['hour_of_day'].plot.hist(bins=np.arange(25)-0.5, ec='black',xticks=np.arange(24), title='Trip count vs hour_of_day')
        #Lets see histograms of number of trips for weekdays vs weekends to analyse the usage pattern in the weekdays and in the weekends.
        #For Weekends
        tmp = self.df_trips[(self.df_trips['day_of_week']==5) | (self.df_trips['day_of_week']==6)]
        plt.figure()
        tmp['hour_of_day'].plot.hist(bins=np.arange(25)-0.5, ec='black',xticks=np.arange(24), title='Trip count vs hour_of_day in weekend')
        #For Weekdays
        tmp = self.df_trips[(self.df_trips['day_of_week']!=5) & (self.df_trips['day_of_week']!=6)]
        plt.figure()
        tmp['hour_of_day'].plot.hist(bins=np.arange(25)-0.5, ec='black',xticks=np.arange(24), title='Trip count vs hour_of_day in weekday')
        """
        To explore how certain feature relates to SHORT and LONG trips
        we compute
        ratioSL = total_number_of_SHORT_trips/total_number_of_SHORT_trips
        and plot for the feature under investigtion
        This should give us an idea how the preference of SHORT and LONG trips change with this feature
        """
        #hour_of_day
        self.plotVsratioSL('hour_of_day', 5)
        #day_of_week
        self.plotVsratioSL('day_of_week', 5)
        #rider_age
        self.plotVsratioSL('rider_age', 5)
        #gender_num
        f = 'gender_num'
        tmp = self.getShortToLongTripCountRatioFor(self.df_trips, f, 5)
        plt.figure()
        sns.barplot(x="gender_num", y="ratioSL", data=tmp)
        ax = plt.gca()
        ax.set_title('ratioSL difference between male(1) and female(0)')
        plt.figure()
        sns.barplot(x="gender_num", y="SHORT", data=tmp)
        ax = plt.gca()
        ax.set_title('SHORT trip counts between male(1) and female(0)')
        plt.figure()
        sns.barplot(x="gender_num", y="LONG", data=tmp)
        ax = plt.gca()
        ax.set_title('LONG trip counts between male(1) and female(0)')
        #from_station_id
        f = 'from_station_id'
        tmp = self.getShortToLongTripCountRatioFor(self.df_trips, f, 5);
        tmp = pd.merge(tmp, self.df_stations, left_on='from_station_id', right_on='id', how='left')
        #NORMALIZE BEFORE........IN THE BEGINING AFTER READING DATA FOR STATIONS.......
        normalized_col = self.normalize(tmp, 'ratioSL')
        tmp['ratioSL_n'] = normalized_col
        plt.figure()
        tmp.plot.scatter(x='long_n', y='lat_n', c='ratioSL_n', s=5, title='Geographial Distribution of ratioSL', colormap='cool', grid=True, xticks=np.arange(0, 1.1, 0.1), yticks=np.arange(0, 1.1, 0.1))
        plt.figure()
        tmp.plot.scatter(x='long_n', y='lat_n', c='SHORT', s=5, title='Geographial Distribution of SHORT trips', colormap='cool', grid=True, xticks=np.arange(0, 1.1, 0.1), yticks=np.arange(0, 1.1, 0.1))
        plt.figure()
        tmp.plot.scatter(x='long_n', y='lat_n', c='LONG', s=5, title='Geographial Distribution of LONG trips', colormap='cool', grid=True, xticks=np.arange(0, 1.1, 0.1), yticks=np.arange(0, 1.1, 0.1))
        
        