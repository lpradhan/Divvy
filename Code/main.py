#Expedia Divvy Challenge...
#Author: Ligaj Pradhan

import os
import Expedia_Divvy_Feature_PreProcessor_N_Explorer as edfp
import Training_And_Validation_Handler as tvh
import Testing_Handler as th
import pickle

parentDataFolder = 'Datasets/Divvy_Trips_2017_Q1Q2'
dataSet_Divvy_Trips_1 = parentDataFolder+'/Divvy_Trips_2017_Q1.csv'
dataSet_Divvy_Trips_2 = parentDataFolder+'/Divvy_Trips_2017_Q2.csv'
dataSet_Divvy_Stations = parentDataFolder+'/Divvy_Stations_2017_Q1Q2.csv'

def prepareData(useOnlyFirstFile):
    absParentFolder = os.path.abspath('..') + '/' + parentDataFolder
    absDivvyTripCSVFile_1 = os.path.abspath('..') + '/'+ dataSet_Divvy_Trips_1
    absDivvyTripCSVFile_2 = os.path.abspath('..') + '/'+ dataSet_Divvy_Trips_2
    absDivvyStationsCSVFile = os.path.abspath('..') + '/' + dataSet_Divvy_Stations
    ed = edfp.DataPreProcessorAndExplorer(absParentFolder, absDivvyTripCSVFile_1, absDivvyTripCSVFile_2, absDivvyStationsCSVFile, useOnlyFirstFile)
    ed.loadDatasets()
    ed.cleanDatasets()
    ed.getInterStationDistancesInKm()
    ed.addGroundTruthColumn()
    ed.extractTimeFeaturesFromDate()
    ed.extractUserFeatures()
    ed.extractStationFeatures()
    ed.exploreFeatures()
    ed.finalizeDataset()
    
def train_N_Validate_Classifier(doFeatureSelection, persistModel):
    absParentFolder = os.path.abspath('..') + '/' + parentDataFolder
    tt = tvh.TrainingAndValidationHandler(absParentFolder)
    tt.load_Train_Valid_Sets()
    print('Feature Correlation without Feature Selection : ')
    tt.showFeatureCorrelations()
    
    #Perform Feature Selection if the 'doFeatureSelection' is on...
    if (doFeatureSelection):
        #tt.featureSelection_with_VarianceThreshold(0.95)
        tt.featureSelection_with_Univariate_Statistical_Tests(9)
        print('Feature Correlation after Feature Selection : ')
        tt.showFeatureCorrelations()
    
    model = tt.trainAndTestWithRandomForest()
    if persistModel:
        if (doFeatureSelection):
            filename = absParentFolder + '/trained_model_with_feature_selection.sav'
        else:
            filename = absParentFolder + '/trained_model.sav'
        
        pickle.dump(model, open(filename, 'wb'))

def test_trained_Classifier(testFeatureSelectedModel):
    #Testing on all test data...
    absParentFolder = os.path.abspath('..') + '/' + parentDataFolder
    t = th.TestingHandler(absParentFolder)
    t.load_Test_Sets()
    t.test_Classifier(testFeatureSelectedModel)
    #Testing in one trip instance...
    #t.getTripCLassForATrip(275,25,1,'Customer',2,20,True)
    
def main():
    prepareData(True)
    train_N_Validate_Classifier(True,True)
    test_trained_Classifier(True)

if __name__== "__main__":
    main()
