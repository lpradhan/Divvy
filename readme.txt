#Expedia Divvy Challenge...
#Author: Ligaj Pradhan


#CLASS DESCRIPTION:

Expedia_Divvy_Feature_PreProcessor_N_Explorer.py 
	This file contains the 'DataPreProcessorAndExplorer' class. 'DataPreProcessorAndExplorer' is  responsible for preparing training, validaiton and test datasets. It also computes physical distance using the longitude and latitude features of stations (with getInterStationDistancesInKm function) and add LONG and SHORT groundtruth values to the dataset. This class also explores features and plots their relationships, which can be very helpful to draw insights about the problem under investigation and data.


Training_And_Validation_Handler.py
	This file contains the 'TrainingAndValidationHandler' class. Everything related to training and valdating is done from 'TrainingAndValidationHandler' class. It has functions to show correlations between features, perform feature selection, report the performance in the validation set and finally perform training using couple of different classifiers (like RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier, SVC, LinearSVC, MLPClassifier and GaussianNB). Depending upon the performace report on the validation set, we can tweak parameters of the classifiers and select differnt features from this class.


Testing_Handler.py
	This class contains the 'TestingHandler' class. Everything related to testing is done from 'TestingHandler' class. 'test_Classifier' function ca load the saved trained model, perform testing on the test dataset and report performance of the classifier on test dataset. 'getTripCLassForATrip' can also classify a single trip description into SHORT or LONG trip.


Main.py 
	This class is the main entry point. In its main function runs three funcitons in sequence.
	1. prepareData(True) 
	2. train_N_Validate_Classifier(True,True) and 
	3. test_trained_Classifier(True)

	prepareData(useOnlyFirstFile) : 
		'DataPreProcessorAndExplorer' is then responsible for preparing training, validaiton and test datasets. 
		'prepareData' function creates an instance of 'DataPreProcessorAndExplorer' class and passes it the parent folder of the datasets as 'absParentFolder' and '2 file names of the Trip datasets' and 'Divvy Stations data file' inside the 'absParentFolder'. 
		Boolean value 'useOnlyFirstFile' will let the 'prepareData' function load only the first data file.
		This class can also explore the dataset and plot  relationships bwetween various features with.  
		'prepareData' function will also save a file called 'Station_Processed_Info.csv' which will contain from_station_id, normalized latitude, normalized longitude, count of long trips from this Station, count of short trips from this Station and ratio between counts of short and long trips from this station using only train data. (This will be appended with from sation id in validation and test datasets as additional station features.)

	train_N_Validate_Classifier(doFeatureSelection, persistModel):
		This function is responsible for training and validation. 
		Boolean value 'doFeatureSelection' allows it to perform feature selection. Boolean value 'persistModel' allows it to save the trained model. 
		This function also plots the correlation between features. 
		It trains a RandomForest classifierwith default 200 decision trees. 
		Everything related to training and valdating is done from 'TrainingAndValidationHandler' class. 
		There are other different classifiers available inside this class which can also be used form this function train_N_Validate_Classifier as well instead of RandomForest.

	test_trained_Classifier(featureSelected):
		This function is responsiblefor testing of the saved trained model with the test dataset.
		It loads the saved model and gets te subset of selected features if boolean parameter 'featureSelected' is true before testing.
		Everything related to testing is done from 'TestingHandler' class. 








