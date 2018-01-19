#Expedia Divvy Challenge...
#Author: Ligaj Pradhan

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class TrainingAndValidationHandler:
    
    def __init__(self, absParentFolder):
        self.absParentFolder = absParentFolder
        self.train_X = pd.DataFrame()
        self.train_Y = pd.DataFrame()
        self.valid_X = pd.DataFrame()
        self.valid_Y = pd.DataFrame()
        
    def load_Train_Valid_Sets(self):        
        train = pd.read_csv(self.absParentFolder + '/train.csv')
        validation = pd.read_csv(self.absParentFolder + '/validation.csv')
        self.train_X = train.loc[:,['lat_n', 'long_n', 'LONG_Trips_FromStation', 'SHORT_Trips_FromStation', 'ratioSL_FromStation', 'rider_age', 'gender_num', 'Customer', 'Dependent', 'Subscriber', 'day_of_week', 'hour_of_day']]
        self.train_Y = train.loc[:,['GroundTruth']].values.ravel()
        self.valid_X = validation.loc[:,['lat_n', 'long_n', 'LONG_Trips_FromStation', 'SHORT_Trips_FromStation', 'ratioSL_FromStation', 'rider_age', 'gender_num', 'Customer', 'Dependent', 'Subscriber', 'day_of_week', 'hour_of_day']]
        self.valid_Y = validation.loc[:,['GroundTruth']].values.ravel()
    
    ###plot a heat map showing correlation between the input features...
    def showFeatureCorrelations(self):
        #Lets see the correlation between features...
        plt.figure()
        sns.heatmap(self.train_X.corr())
        ax = plt.gca()
        ax.set_title('Correlation between input features')
    
    ###Not dependent on the target variable...
    def featureSelection_with_VarianceThreshold(self,th):
        #Using VarianceThreshold...
        print('Columns before feature selection:')
        print(self.train_X.columns.values)
        sel = VarianceThreshold(threshold=(th * (1 - th)))
        sel.fit(self.train_X)
        self.train_X = self.train_X.iloc[:, sel.get_support()]
        self.valid_X = self.valid.iloc[:, sel.get_support()]
        print('Selected features after feature selection with VarianceThreshold:')
        print(self.train_X.columns.values)
    
    ###Dependent on the target variable...
    def featureSelection_with_Univariate_Statistical_Tests(self, numOfFeaturesToSelect):
        #Using VarianceThreshold...
        print('Columns before feature selection:')
        print(self.train_X.columns.values)
        sel = SelectKBest(chi2, k=numOfFeaturesToSelect).fit(self.train_X, self.train_Y)
        self.train_X = self.train_X.iloc[:, sel.get_support()]
        self.valid_X = self.valid_X.iloc[:, sel.get_support()]
        print('Selected features after feature selection with Univariate Statistical Tests:')
        print(self.train_X.columns.values)
    
    ###Provides accuracy in percentage, confusin matrix and the precision:recall:f1 measures...
    def reportPerformance(self, c):
        print(self.valid_X.info())
        predictions = c.predict(self.valid_X)
        print('Validation Test Accuracy: '+ str(accuracy_score(self.valid_Y, predictions)*100) + ' %')
        cm = confusion_matrix(self.valid_Y, predictions)
        print('confusion_matrix : ' )
        print(cm)
        print(classification_report(self.valid_Y, predictions))
        predictions = c.predict(self.train_X)
        print('Training Accuracy: '+ str(accuracy_score(self.train_Y, predictions)))
    
    ###use the trained classifier c to output the relative importance of different features used for learning the classification model...  
    def reportFeatureImportances(self, c):
        importances = c.feature_importances_
        std = np.std([tree.feature_importances_ for tree in c.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")
        for f in range(self.train_X.values.shape[1]):
            print("%d. %s (feature %d) (%f)" % (f + 1, self.train_X.columns.values[indices[f]], indices[f], importances[indices[f]]))
        # Plotting the feature importance...
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(self.train_X.shape[1]), importances[indices], color="g", yerr=std[indices], align="center")
        plt.xticks(range(self.train_X.shape[1]), indices)
        plt.xlim([-1, self.train_X.shape[1]])
        plt.show()
        
    def trainAndTestWithRandomForest(self):
        print('Training in progress...with RandomForestClassifier')
        c =  RandomForestClassifier(n_estimators=200)
        c.fit(self.train_X, self.train_Y)
        self.reportPerformance(c)
        self.reportFeatureImportances(c)
        print('Training finished...with RandomForestClassifier')
        return c
        
        
    def trainAndTestWithExtraTreesClassifier(self):
        print('Training in progress...with ExtraTreesClassifier')
        c =  ExtraTreesClassifier(n_estimators=200)
        c.fit(self.train_X, self.train_Y)
        self.reportPerformance(c)
        self.reportFeatureImportances(c)
        print('Training finished...with ExtraTreesClassifier')
        return c
    
    def trainAndTestWithDecisionTree(self):
        print('Training in progress...with DecisionTreeClassifier')
        c =  DecisionTreeClassifier()
        c.fit(self.train_X, self.train_Y)
        self.reportPerformance(c)
        print('Training finished...with DecisionTreeClassifier')
        return c
        
    def trainAndTestWithSVC(self):
        print('Training in progress...with SVC')
        c =  SVC(max_iter=10000, tol=1e-4)
        c.fit(self.train_X, self.train_Y)
        self.reportPerformance(c)
        print('Training finished...with SVC')
        return c
        
    def trainAndTestWithLinearSVC(self):
        print('Training in progress...with LinearSVC')
        c =  LinearSVC(max_iter=10000, tol=1e-4)
        c.fit(self.train_X, self.train_Y)
        self.reportPerformance(c)
        print('Training finished...with LinearSVC')
        return c
        
    def trainAndTestWithMLP(self):
        print('Training in progress...with MLPClassifier')
        c =  MLPClassifier(max_iter =10000, tol=1e-4, alpha=1e-5, hidden_layer_sizes=(7, 7), random_state=1)
        c.fit(self.train_X, self.train_Y)
        self.reportPerformance(c)
        print('Training finished...with MLPClassifier')
        return c
        
    def trainAndTestWithNB(self):
        print('Training in progress...with GaussianNB')
        c =  GaussianNB()
        c.fit(self.train_X, self.train_Y)
        self.reportPerformance(c)
        print('Training finished...with GaussianNB')
        return c
