#Expedia Divvy Challenge...
#Author: Ligaj Pradhan

import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class TestingHandler:
    
    def __init__(self, absParentFolder):
        self.absParentFolder = absParentFolder
        self.test_X = pd.DataFrame()
        self.test_Y = pd.DataFrame() 
    
    ###load test.csv file produced by the 'DataPreProcessorAndExplorer' class
    def load_Test_Sets(self):        
        test = pd.read_csv(self.absParentFolder + '/test.csv')
        self.test_X = test.loc[:,['lat_n', 'long_n', 'LONG_Trips_FromStation', 'SHORT_Trips_FromStation', 'ratioSL_FromStation', 'rider_age', 'gender_num', 'Customer', 'Dependent', 'Subscriber', 'day_of_week', 'hour_of_day']]
        self.test_Y = test.loc[:,['GroundTruth']].values.ravel()
    
    ###Provides accuracy in percentage, confusin matrix and the precision:recall:f1 measures...
    def reportPerformance(self, c):
        predictions = c.predict(self.test_X)
        print('Testing Accuracy: '+ str(accuracy_score(self.test_Y, predictions)*100) + ' %')
        cm = confusion_matrix(self.test_Y, predictions)
        print('confusion_matrix : ' )
        print(cm)
        print(classification_report(self.test_Y, predictions))

    ###Test is performed in the whole test dataset
    def test_Classifier(self, featureSelected):
        if featureSelected:
            filename = self.absParentFolder + '/trained_model_with_feature_selection.sav'
            #These Features were selected during feature selection while training.....So we load only them
            self.test_X = self.test_X.loc[:,['lat_n', 'long_n', 'LONG_Trips_FromStation', 'SHORT_Trips_FromStation', 'ratioSL_FromStation', 'rider_age', 'gender_num', 'day_of_week', 'hour_of_day']]
        else:
            filename = self.absParentFolder + '/trained_model.sav'
        if (os.path.isfile(filename)):
            trained_model = pickle.load(open(filename, 'rb'))
            self.reportPerformance(trained_model)
        else:
            print('Trained model not available...')
    
    ###Test for single trip instance described in its parameters...
    def getTripCLassForATrip(self, fromStationId, rider_age, gender, usertype, day_of_week, hour_of_day, modelWithFeatureSelection):
        df_stations = pd.read_csv(self.absParentFolder + '/Station_Processed_Info.csv')
        vals = df_stations[df_stations['from_station_id']==fromStationId].loc[:,['LONG_Trips_FromStation','SHORT_Trips_FromStation','ratioSL_FromStation','lat_n','long_n',]].values.tolist()
        # Get preprocessed features for from_station_id which was computed from training data
        LONG_Trips_FromStation = vals[0][0]
        SHORT_Trips_FromStation = vals[0][1]
        ratioSL_FromStation = vals[0][2]
        lat_n = vals[0][3]
        long_n = vals[0][4]
        gender_num = 1 if (gender=='Male') else 0
        Customer = '1' if (usertype=='Customer') else 0
        Subscriber = 1 if (usertype=='Subscriber') else 0
        Dependent = 1 if (usertype=='Dependent') else 0
        # Load trained classifier model...
        if (modelWithFeatureSelection):
            filename = self.absParentFolder + '/trained_model_with_feature_selection.sav'
            testDF = pd.DataFrame([[lat_n, long_n, LONG_Trips_FromStation, SHORT_Trips_FromStation, ratioSL_FromStation, rider_age, gender_num, day_of_week, hour_of_day]], columns=['lat_n', 'long_n', 'LONG_Trips_FromStation', 'SHORT_Trips_FromStation', 'ratioSL_FromStation', 'rider_age', 'gender_num', 'day_of_week', 'hour_of_day'])
        else:
            filename = self.absParentFolder + '/trained_model.sav'
            testDF = pd.DataFrame([[lat_n, long_n, LONG_Trips_FromStation, SHORT_Trips_FromStation, ratioSL_FromStation, rider_age, gender_num, Customer, Subscriber, Dependent, day_of_week, hour_of_day]], columns=['lat_n', 'long_n', 'LONG_Trips_FromStation', 'SHORT_Trips_FromStation', 'ratioSL_FromStation', 'rider_age', 'gender_num', 'Customer', 'Dependent', 'Subscriber', 'day_of_week', 'hour_of_day'])
        if (os.path.isfile(filename)):
            loaded_model = pickle.load(open(filename, 'rb'))
            result = loaded_model.predict(testDF)
            tripClass = 'SHORT' if result[0]==1 else 'LONG'
            print('The class for this single trip instance is : ' + tripClass)
            return tripClass
        else:
            print('Trained model not available...')