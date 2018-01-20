#Expedia Divvy Challenge...
#Author: Ligaj Pradhan

In this project a random forest classifier is trained to classify bike trips from divvy dataset into SHORT or LONG trips. Only thse features that are know before a trip is satrted is used or training the model. Some of the packages you need to run the code are pandas, matplotlib, sklearn, seaborn and pickle.

To execute simply run the Main.py file.



__________________________________________________
Main.py 
__________________________________________________
This class is the main entry point. Execute this file to prepare the training, validation and test data and perform training, validation and testing. In its main function it runs the following three funcitons in sequence.
1. prepareData(True) 
2. train_N_Validate_Classifier(True,True) and 
3. test_trained_Classifier(True)

Note: 
1. Station_Proecssed_Info.csv, train.csv, validation.csv, test.csv are created in the first step i.e. prepareData(True)  
2. trained_model_with_feature_selection.sav if rain_N_Validate_Classifier(True,True) or trained_model.csv if rain_N_Validate_Classifier(False,True) is created in the second step
3. If you want to re-run steps 1 and 2 please delete the corresponding files created at each step from the dataset folder.
4. If you have already created train, validation and test datasets, you can comment step 1 i.e. prepareData(True), to try step 2 i.e. training and validation repeatedly.
5. Similarly, if step 1 and 2 are already completed and you only want to run testing on the test data, comment out the first two steps.
_____________________________________
prepareData(useOnlyFirstFile) : 
_____________________________________
'DataPreProcessorAndExplorer' is then responsible for preparing training, validaiton and test datasets. 
'prepareData' function creates an instance of 'DataPreProcessorAndExplorer' class and passes it the parent folder of the datasets as 'absParentFolder' and '2 file names of the Trip datasets' and 'Divvy Stations data file' inside the 'absParentFolder'. 
Boolean value 'useOnlyFirstFile' will let the 'prepareData' function load only the first data file.
This class can also explore the dataset and plot  relationships bwetween various features with.  
'prepareData' function will also save a file called 'Station_Processed_Info.csv' which will contain from_station_id, normalized latitude, normalized longitude, count of long trips from this Station, count of short trips from this Station and ratio between counts of short and long trips from this station using only train data. (This will be appended with from sation id in validation and test datasets as additional station features.)
______________________________________________________________
train_N_Validate_Classifier(doFeatureSelection, persistModel):
______________________________________________________________
This function is responsible for training and validation. 
Boolean value 'doFeatureSelection' allows it to perform feature selection. Boolean value 'persistModel' allows it to save the trained model. 
This function also plots the correlation between features. 
It trains a RandomForest classifierwith default 200 decision trees. 
Everything related to training and valdating is done from 'TrainingAndValidationHandler' class. 
There are other different classifiers available inside this class which can also be used form this function train_N_Validate_Classifier as well instead of RandomForest.
__________________________________________
test_trained_Classifier(featureSelected):
__________________________________________
This function is responsiblefor testing of the saved trained model with the test dataset.
It loads the saved model and gets te subset of selected features if boolean parameter 'featureSelected' is true before testing.
Everything related to testing is done from 'TestingHandler' class. 




#CLASS DESCRIPTIONS:

__________________________________________________
Expedia_Divvy_Feature_PreProcessor_N_Explorer.py 
__________________________________________________
	This file contains the 'DataPreProcessorAndExplorer' class. 'DataPreProcessorAndExplorer' is  responsible for preparing training, validaiton and test datasets. It also computes physical distance using the longitude and latitude features of stations (with getInterStationDistancesInKm function) and add LONG and SHORT groundtruth values to the dataset. This class also explores features and plots their relationships, which can be very helpful to draw insights about the problem under investigation and data.

__________________________________________________
Training_And_Validation_Handler.py
__________________________________________________
	This file contains the 'TrainingAndValidationHandler' class. Everything related to training and valdating is done from 'TrainingAndValidationHandler' class. It has functions to show correlations between features, perform feature selection, report the performance in the validation set and finally perform training using couple of different classifiers (like RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier, SVC, LinearSVC, MLPClassifier and GaussianNB). Depending upon the performace report on the validation set, we can tweak parameters of the classifiers and select differnt features from this class.

__________________________________________________
Testing_Handler.py
__________________________________________________
	This class contains the 'TestingHandler' class. Everything related to testing is done from 'TestingHandler' class. 'test_Classifier' function ca load the saved trained model, perform testing on the test dataset and report performance of the classifier on test dataset. 'getTripCLassForATrip' can also classify a single trip description into SHORT or LONG trip.

__________________________________________________









