#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 23:06:00 2017

@author: Kenneth Chen
"""

# modules import
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import sklearn
import IPython
import timeit
import random
import time

# Version check
print('All libraries versions\n')
print('Python version: {}'.format(sys.version))
print('pandas version: {}'.format(pd.__version__))
print('NumPy versino: {}'.format(np.__version__))
print('matplotlib version: {}'.format(matplotlib.__version__))
print('SciPy version: {}'.format(sp.__version__))
print('IPython version: {}'.format(IPython.__version__))
print('scikit-learn version: {}\n'.format(sklearn.__version__))

# Listing the input files
from subprocess import check_output
print('List of files available\n')
print(check_output(['ls', '/Users/lwinchen/Desktop/Machine Learning/Kaggle/Titanic']).decode('utf8'))

# Model Algorithms attributes (GLENN ST D)
from sklearn import gaussian_process, linear_model, ensemble, neighbors, naive_bayes, svm, tree, discriminant_analysis

# Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection, model_selection, metrics

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt   # Done in the beginning, but just to be consistent with the learning process
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

# view graph in Jupyter Notebook
# matplotlib inline

mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

# load the data
data_train_ori = pd.read_csv('/Users/lwinchen/Desktop/Machine Learning/Kaggle/Titanic/train.csv')
data_test_ori = pd.read_csv('/Users/lwinchen/Desktop/Machine Learning/Kaggle/Titanic/test.csv')

# copying the original data before manipulating the data
data_train = data_train_ori.copy()
data_test = data_test_ori.copy()
data_test_id = data_test['PassengerId']

# prior view of the data to check
# min and max to make sure if the range is reasonable, eg, if the age is 800 or -30
print(data_train.head(5))  # printing the first 5 rows

print('\nPlease check each variable/feature traits as in min, max, mean\n')
print(data_train.describe()) # training data information
# After checking the data, see if they have null values

print('\nTraining data with null values\n')
print(data_train.isnull().sum())

print('-'*10)
print('\nTest data with null values\n')
print(data_test.isnull().sum())

# Making a group so as to update two data at once in later steps
data_all = [data_train, data_test]

print('\nSince we see null values in Age, Embarked, Fare, Cabin variables, we are going to fill up the null value. However, as some variables are not important, we are going to ignore.\n')

# Creating 'Title' from the Name feature
for dataset in data_all:
    dataset['Title'] = dataset['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    
print('\nA new column Title was created. There are more men than women on board.\n')
print(data_train['Title'].value_counts())

# creating a new feature 'NameLen' to describe the length of the name
for dataset in data_all:
    dataset['NameLen'] = dataset['Name'].apply(lambda x: len(x))
    
print('\nLooking at the Title, we see there are Titles whose population is just one. Eg, Sir. In order to put all rare Titles into one, we create Misc category.\n')
# First we need to assign some kind of value in Title, to combine those rare Titles. 
for dataset in data_all:
    title_names = (dataset['Title'].value_counts() < 10)
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    print(dataset['Title'].value_counts())
    print('-'*10)
    
# ------------------- Fill Empty Age based on Sex, Pclass and Title ---------
# complete the null in Age variable with the 'Title' median
# This will fill up null value in Age with their respective Title. 
# Eg, Mr. Kenneth whose age is missing, will be filled up with the median age of all Mr. persons, rather than the median age of all people on board
grouped_train = data_train.groupby(['Sex', 'Pclass', 'Title'])
grouped_median_train = grouped_train.median()

grouped_test = data_test.groupby(['Sex', 'Pclass', 'Title'])
grouped_median_test = grouped_test.median()

def process_age():
    
    global data_all
    
    def fillAges(row, grouped_median):
        if row['Sex'] == 'female' and row['Pclass'] == 1:   # writing in another form (row.Sex == 'female')
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 1, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 1, 'Mrs']['Age']
            elif row['Title'] == 'Misc':
                return grouped_median.loc['female', 1, 'Misc']['Age']
        
        elif row['Sex'] == 'female' and row['Pclass'] == 2:   # writing in another form (row.Sex == 'female')
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 2, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 2, 'Mrs']['Age']
            elif row['Title'] == 'Misc':
                return grouped_median.loc['female', 2, 'Misc']['Age']
    
        elif row['Sex'] == 'female' and row['Pclass'] == 3:   # writing in another form (row.Sex == 'female')
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 3, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 3, 'Mrs']['Age']
            elif row['Title'] == 'Misc':
                return grouped_median.loc['female', 3, 'Misc']['Age']
    
        elif row['Sex'] == 'male' and row['Pclass'] == 1:   # writing in another form (row.Sex == 'female')
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 1, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 1, 'Mr']['Age']
            elif row['Title'] == 'Misc':
                return grouped_median.loc['male', 1, 'Misc']['Age']

        elif row['Sex'] == 'male' and row['Pclass'] == 2:   # writing in another form (row.Sex == 'female')
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 2, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 2, 'Mr']['Age']
            elif row['Title'] == 'Misc':
                return grouped_median.loc['male', 2, 'Misc']['Age']
            
        elif row['Sex'] == 'male' and row['Pclass'] == 3:   # writing in another form (row.Sex == 'female')
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 3, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 3, 'Mr']['Age']
            elif row['Title'] == 'Misc':
                return grouped_median.loc['male', 3, 'Misc']['Age']    
    
    data_train.Age = data_train.apply(lambda r: fillAges(r, grouped_median_train) if np.isnan(r['Age']) else r['Age'], axis = 1)
    data_test.Age = data_test.apply(lambda r: fillAges(r, grouped_median_test) if np.isnan(r['Age']) else r['Age'], axis = 1)
    
    print('Age filling up ok.')
    
process_age()

            
for dataset in data_all:
    # dataset.Age.fillna(dataset.groupby('Title').Age.transform('median'), inplace=True)
    dataset.Fare.fillna(dataset.groupby('Pclass').Fare.transform('median'), inplace=True)
    
    # dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    
print('\nChecking null values after fillup.\n')
print(data_train.isnull().sum())
print('-'*10)
print(data_test.isnull().sum())

# ------------------- A new feature 'Surname' ------------------------------- 
# Create a new feature 'Surname' from the 'Name' feature
for dataset in data_all:
    dataset['Surname'] = dataset['Name'].str.split(', ', expand=True)[0]

 
group_by_surname = data_train.groupby(['Survived', 'Surname'])
survived_surname = group_by_surname.size().unstack()

gpsurname_train = data_train.groupby(['Surname', 'Pclass'])
gpsurname_train_mean = gpsurname_train.mean()

gpsurname_test = data_test.groupby(['Surname', 'Pclass'])
gpsurname_test_mean = gpsurname_test.mean()

surname_intest = list(set(data_test['Surname']) - set(data_train['Surname']))


#def fillBySurname(row, gp_mean):

#    global data_train, data_test
#    print('This is Surname: ', row['Surname'],'And this is the Pclass: ', row['Pclass'])
#    print('This is the mean survived: ', gp_mean.loc[row['Surname'], row['Pclass']]['Survived'])
#    if gp_mean.loc[row['Surname'], row['Pclass']]['Survived'] == 0.5:
#        return 0.5
#    elif gp_mean.loc[row['Surname'], row['Pclass']]['Survived'] < 0.5:
#        return 0
#    else:
#        return 1
    
#data_train['Survived_by_surname'] = data_train.apply(lambda r: fillBySurname(r, gpsurname_train_mean), axis = 1)
# data_test['Survived_by_surname'] = data_test.apply(lambda r: fillBySurname(r, gpsurname_train_mean), axis = 1)

# for data_train and data_test

def fillBySurname(row, gp_mean):
    
    global data_train, data_test, surname_intest
    
    if row['Surname'] in surname_intest:
        return 0.5
    else:
        pcl = list(data_train.loc[data_train['Surname'] == row['Surname'], 'Pclass'])
        
    if row['Pclass'] in pcl: 
        print('This is Surname: ', row['Surname'],'And this is the Pclass: ', row['Pclass'])
        print('This is the mean survived: ', gp_mean.loc[row['Surname'], row['Pclass']]['Survived'])
        if gp_mean.loc[row['Surname'], row['Pclass']]['Survived'] == 0.5:
            return 0.5
        elif gp_mean.loc[row['Surname'], row['Pclass']]['Survived'] < 0.5:
            return 0
        else:
            print('1')
            return 1
    else:
        return 0.5
        
data_train['Survived_by_surname'] = data_train.apply(lambda r: fillBySurname(r, gpsurname_train_mean), axis = 1)
data_test['Survived_by_surname'] = data_test.apply(lambda r: fillBySurname(r, gpsurname_train_mean), axis = 1)


# ------------------- SocioEco Feature Engineering --------------------------    
# Create a new feature 'SocioEco' based on sex, fare and pclass
def SEstatus(row):
    if (row.Sex == 'female') and (row.Fare > 80) and (row.Pclass == 1 and 2):
        return 1
    elif (row.Sex == 'female') and (row.Fare > 80) and (row.Pclass == 3):
        return 2
    elif (row.Sex == 'female') and (row.Fare < 80 and row.Fare >20):
        return 2
    elif (row.Sex == 'female') and (row.Fare < 20):
        return 3
    elif (row.Sex == 'male') and (row.Fare > 300) and (row.Pclass == 1):
        return 1
    elif (row.Sex == 'male') and (row.Fare > 300) and (row.Pclass == 2 and 3):
        return 2
    elif (row.Sex == 'male') and (row.Fare < 300 and row.Fare > 80) and (row.Pclass == 1 and 2):
        return 2
    elif (row.Sex == 'male') and (row.Fare < 300 and row.Fare > 80) and (row.Pclass == 3):
        return 3
    elif (row.Sex == 'male') and (row.Fare < 80 and row.Fare >20):
        return 3
    else:
        return 4
    
for dataset in data_all:
    dataset.loc[:, 'SocioEco'] = dataset.apply(SEstatus, axis=1)
    
# Grouping 
group_by_socioeco = data_train.groupby(['Survived', 'SocioEco'])
survived_socioeco = group_by_socioeco.size().unstack()
print(survived_socioeco)
group_by_sex_socioeco = data_train.groupby(['Sex', 'SocioEco'])
sex_socioeco = group_by_sex_socioeco.size().unstack()
print(sex_socioeco)
# survived_socioeco.plot.bar()


# ------------------- End ---------------------------------------------------

# ------------------- Dropping Unwanted Features ----------------------------
# Delete unwanted features 
for dataset in data_all:
    dataset.drop(['PassengerId', 'Cabin', 'Ticket', 'Embarked'], axis = 1, inplace=True)
    
print('\nPassengerId, Cabin, Embarked and Ticket features are dropped.\n')
print('\nChecking all info again.\n')
print(data_train.info())
print('\nChecking null values.\n')
print(data_train.isnull().sum())
print('-'*10)
print(data_test.isnull().sum())
print('\nData are now all clean without null values.\n')
    
# Feature Engineeering for some algorithms analysis such as Decision Trees
# First, we need to discretize Family count, either with family or alone
# Second, Age is continuous, and needs to be grouped as manageable as possible in 4 groups
# Third, Fare is also continous, and needs to be grouped.
print('\nNew columns such as FamilySize, AgeBin, FareBin are created in order to group continous data.\n')
for dataset in data_all:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0   # This avoids the false warning of copying
    dataset.loc[dataset['FamilySize'] > 4, 'IsAlone'] = 2   # This creates any family with more than 4 members will be categorized as IsAlone because their survival is almost similar to single.
    
    dataset['AgeBin'] = pd.cut(dataset['Age'], 8)
    dataset['FareBin'] = pd.cut(dataset['Fare'], 10)
    dataset['NameLenBin'] = pd.cut(dataset['NameLen'].astype(int), 5)
    
print(data_train.info())
print('-'*10)
print(data_test.info())

# Quick check at the survival outcome between Age, Fare, and Family Size
print('\nAfter grouping continous data into a manageable group for IsAlone, AgeBin, FareBin, we are going to check if there are any correlation between those variables and survival outcome.\n')

group_by_family = data_train.groupby(['Survived', 'IsAlone'])
group_by_sibsp = data_train.groupby(['Survived', 'SibSp'])
group_by_parch = data_train.groupby(['Survived', 'Parch'])
group_by_age = data_train.groupby(['Survived', 'AgeBin'])
group_by_fare = data_train.groupby(['Survived', 'FareBin'])
group_by_sex = data_train.groupby(['Survived', 'Sex'])

# Unstacking the groups 
survived_family = group_by_family.size().unstack()
survived_sibsp = group_by_sibsp.size().unstack()
survived_parch = group_by_parch.size().unstack()
survived_age = group_by_age.size().unstack()
survived_fare = group_by_fare.size().unstack()
survived_sex = group_by_sex.size().unstack()

# Looking at the bar graph
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6,10))
plt.subplots_adjust(wspace = 0.3, hspace= 0.3);
survived_family.plot.bar(ax=axes[0,0]); axes[0,0].set_title('Family Size');
survived_age.plot.bar(ax=axes[0,1]); axes[0,1].set_title('Age');
survived_fare.plot.bar(ax=axes[1,0]); axes[1,0].set_title('Fare');
survived_sex.plot.bar(ax=axes[1,1]); axes[1,1].set_title('Sex');
plt.show()

print('\nLooking at the bar graph, there is no stark difference between Survival and Age. However, with regards to Family size, Fare and Sex, it shows a clear pattern that passengers who came with a family or who paid a higher fare were more likely to survive as well as the sex of the passenger.\n')
print('\nWe will again look at those features with different graphs by seaborn.\n')

# Seaborn graphics
fig, axes = plt.subplots(2, 2, figsize=(6, 10))
sns.barplot(x='FamilySize', y='Survived', data=data_train, ax = axes[0,0])
sns.barplot(x='AgeBin', y='Survived', data=data_train, ax=axes[0,1])
sns.barplot(x='FareBin', y='Survived', data=data_train, ax=axes[1,0])
sns.barplot(x='Sex', y='Survived', data=data_train, ax=axes[1,1])
plt.show()

print('\nLooking at those bar graphs shows a clear patten that Family size with 4 had a higher chance of survival whereas Age did not seem to matter a lot. Passengers who paid higher fare were more likely to survive. Besides, sex of the passenger also mattered a lot in survival in Titanic disaster.\n')

# Converting variables into 0, 1, 2, ... for algorithms analysis in later steps as in Decision trees
# More refined than features engineering because of 0, 1, 2, ... encoded digits
le = LabelEncoder()
for dataset in data_all:
    dataset['Sex_led'] = le.fit_transform(dataset['Sex'])
    dataset['Title_led'] = le.fit_transform(dataset['Title'])
    dataset['AgeBin_led'] = le.fit_transform(dataset['AgeBin'])
    dataset['FareBin_led'] = le.fit_transform(dataset['FareBin'])
    dataset['NameLenBin_led'] = le.fit_transform(dataset['NameLenBin'])
    dataset['SocioEco_led'] = le.fit_transform(dataset['SocioEco'])
    dataset['IsAlone_led'] = le.fit_transform(dataset['IsAlone'])
    dataset['Survived_by_surname_led'] = le.fit_transform(dataset['Survived_by_surname'])
    
Target = ['Survived']


# Define x variables for original features
# This will help 'for loop' function instead of calling all the features individually in later step
data_train_x = ['Sex', 'Pclass', 'SocioEco', 'Title', 'Surname', 'Survived_by_surname', 'NameLen', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']
data_train_xcalc = ['Sex_led', 'Pclass', 'SocioEco_led', 'Title_led','Survived_by_surname_led', 'NameLenBin_led', 'SibSp', 'Parch', 'AgeBin_led', 'FareBin_led', 'FamilySize', 'IsAlone_led']
data_train_xy = Target + data_train_x

# Define x variables without continuous data
data_train_xled = ['Sex_led', 'Pclass', 'SocioEco_led', 'Title_led', 'Survived_by_surname_led', 'NameLenBin_led', 'AgeBin_led', 'FareBin_led', 'FamilySize', 'IsAlone_led']
data_train_xledy = Target + data_train_xled

# Making Dummy on my own called data_train_dummy (dtd)
dtd_sex = pd.get_dummies(data_train['Sex_led'], prefix='Sex')
dtd_pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
dtd_socioeco = pd.get_dummies(data_train['SocioEco_led'], prefix='SocioEco')
dtd_title = pd.get_dummies(data_train['Title_led'], prefix='Title')
dtd_namelen = pd.get_dummies(data_train['NameLenBin_led'], prefix='NameLen')
dtd_age = pd.get_dummies(data_train['AgeBin_led'], prefix='Age')
dtd_fare = pd.get_dummies(data_train['FareBin_led'], prefix='Fare')
dtd_family = pd.get_dummies(data_train['FamilySize'], prefix='FamilySize')
dtd_isalone = pd.get_dummies(data_train['IsAlone_led'], prefix='IsAlone')

data_train_dummy = pd.concat([dtd_sex, dtd_pclass, dtd_socioeco, dtd_title, dtd_namelen, dtd_age, dtd_fare,  dtd_family, dtd_isalone], axis = 1)

# Test dummy
dttd_sex = pd.get_dummies(data_test['Sex_led'], prefix='Sex')
dttd_pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
dttd_socioeco = pd.get_dummies(data_test['SocioEco_led'], prefix='SocioEco')
dttd_title = pd.get_dummies(data_test['Title_led'], prefix='Title')
dttd_namelen = pd.get_dummies(data_test['NameLenBin_led'], prefix='NameLen')
dttd_age = pd.get_dummies(data_test['AgeBin_led'], prefix='Age')
dttd_fare = pd.get_dummies(data_test['FareBin_led'], prefix='Fare')
dttd_family = pd.get_dummies(data_test['FamilySize'], prefix='FamilySize')
dttd_isalone = pd.get_dummies(data_test['IsAlone_led'], prefix='IsAlone')

data_test_dummy = pd.concat([dttd_sex, dttd_pclass, dttd_socioeco, dttd_title, dttd_namelen, dttd_age, dttd_fare,  dttd_family, dttd_isalone], axis = 1)



# Define x and y variables for dummy features
# Dummy varibles like Boolean in astype(int)
# data_train_dummy = pd.get_dummies(data_train[data_train_xled])
data_train_x_dummy = data_train_dummy.columns.tolist()
data_train_xy_dummy = Target + data_train_x_dummy
print('\nDummy X Y: ', data_train_xy_dummy, '\n')

# Split data into train and test sub-data for cross-validation purposes
train_x, test_x, train_y, test_y = model_selection.train_test_split(data_train[data_train_xcalc], data_train[Target], random_state = 0)
train_xled, test_xled, train_yled, test_yled = model_selection.train_test_split(data_train[data_train_xled], data_train[Target], random_state = 10)
train_xdummy, test_xdummy, train_ydummy, test_ydummy = model_selection.train_test_split(data_train_dummy[data_train_x_dummy], data_train[Target], random_state = 20)

print('\ndata_train shape: {}'.format(data_train.shape))
print('\ntrain_x shape: {}'.format(train_x.shape))
print('\ntest_x shape: {}'.format(test_x.shape))

# Checking each feature against Survival as long as the feature is not float64
for x in data_train_x:
    if data_train[x].dtype != 'float64':
        print('Survival correlation by: ', x)
        print(data_train[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')
        
# Heatmap for all features 
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize=(10, 10))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(df.corr(), cmap = colormap, square=True, cbar_kws={'shrink':.9}, ax=ax, annot=True, linewidths=0.1, vmax=1.0, linecolor='white', annot_kws={'fontsize':10})
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    plt.show()
    
correlation_heatmap(data_train)


# -------------- Data Analysis by ML Algorithms --------------
# GLENN ST D
MLA = [
       # Gaussian Processes
       gaussian_process.GaussianProcessClassifier(),
       
       # Linear Model
       linear_model.LogisticRegressionCV(),
       linear_model.PassiveAggressiveClassifier(),
       linear_model.RidgeClassifierCV(),
       linear_model.SGDClassifier(),
       linear_model.Perceptron(),
       
       # Ensemble Methods
       ensemble.AdaBoostClassifier(),
       ensemble.BaggingClassifier(),
       ensemble.ExtraTreesClassifier(),
       ensemble.GradientBoostingClassifier(),
       ensemble.RandomForestClassifier(),
       
       # Naive Bayes
       naive_bayes.BernoulliNB(),
       naive_bayes.GaussianNB(),
       
       # Nearest Neighbor
       neighbors.KNeighborsClassifier(),
       
       # SVM
       svm.SVC(probability=True),
       svm.NuSVC(probability=True),
       svm.LinearSVC(),
       
       # Trees
       tree.DecisionTreeClassifier(),
       tree.ExtraTreeClassifier(),
       
       # Discriminant Analysis
       discriminant_analysis.LinearDiscriminantAnalysis(),
       discriminant_analysis.QuadraticDiscriminantAnalysis()
       
       ]

# Splitting data_test into 60/30 with random subset of data
cv_split = model_selection.ShuffleSplit(n_splits=10, train_size = 0.7, test_size = 0.2, random_state = 0)

# Create MLA metrics to compare
MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)
MLA_predict = data_train[Target]

# ML Algorithms analysis and save performance in MLA_compare table
# Use data_train[Target].values.ravel in cross_validate and fit because there's a false positive warning if you don't.

row_index = 0
for alg in MLA:
    
    MLA_name = alg.__class__.__name__
    
    cv_results = model_selection.cross_validate(alg, data_train[data_train_xled], data_train[Target].values.ravel(), cv = cv_split, return_train_score=True)
        
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = (cv_results['train_score'].mean())*100
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = (cv_results['test_score'].mean())*100
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3
    MLA_compare.loc[row_index, 'MLA Time to fit train data'] = cv_results['fit_time'].mean()
    
    # Save MLA predictions
    alg.fit(data_train_dummy, data_train[Target].values.ravel())
    MLA_predict[MLA_name] = alg.predict(data_train_dummy)
    
    row_index += 1
    
# Sorting MLA_compare table
MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace = True)
print(MLA_compare)

sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=MLA_compare, color='b')
plt.title('\nMachine Learning Algorithms Accuracy Score.\n')
plt.xlabel('Accuracy Score (%)\n')
plt.ylabel('Algorithms\n')
plt.show()

# ------------ Decision Tree (DT) with or without hyperparameters -------------
# without hyperparamters, DT algorithm has been already tested beforehand. But here we will try it again just to be clear with a single algorithm performance
dtree = tree.DecisionTreeClassifier(random_state = 20)
dt_results = model_selection.cross_validate(dtree, data_train[data_train_xled], data_train[Target].values.ravel(), cv = cv_split, return_train_score=True)
dtree.fit(data_train[data_train_xled], data_train[Target].values.ravel())

print('DT parameters used: ', dtree.get_params())
print('DT train score mean: {}'.format(dt_results['train_score'].mean()*100))
print('DT test score mean: {}'.format(dt_results['test_score'].mean()*100))
print('DT test score std: {}'.format(dt_results['test_score'].std()*100*3))
print('-'*10)

# setting up hyperparameters for DT algorithms
param_grid = {'criterion':['gini', 'entropy'], 'max_depth':[2, 4, 6, 8, 10, None], 'random_state':[0], 'max_features':[5, 6, 7, 8], 'min_samples_leaf':[1, 3, 10]}
tune_model = model_selection.GridSearchCV(dtree, param_grid=param_grid, scoring='roc_auc', cv=cv_split, return_train_score=True)
tune_model.fit(data_train[data_train_xled], data_train[Target].values.ravel())

print('DT parameters used: ', tune_model.best_params_)
print('DT train score mean with hyperparameters: {}'.format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('DT test score mean with hyperparameters: {}'.format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('DT test score std with hyperparameters: +/- {}'.format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('-'*10)

# tune model with Feature Selection (Recursive Feature Elimination RFE)
print('DT train shape before RFE: ', data_train[data_train_xled].shape)
print('DT train columns before RFE: ', data_train[data_train_xled].columns.values)
print('DT train score mean before RFE: {}'.format(dt_results['train_score'].mean()*100))
print('DT test score mean before RFE: {}'.format(dt_results['test_score'].mean()*100))
print('DT test score std before REF: +/- {}'.format(dt_results['test_score'].std()*100*3))
print('-'*10)

# features selection
# Since this is a feature selection, you don't need to modify the decision tree estimator/classifier, you just need to modify the list of features.
# However instead of manually selecting features, you'd use RFE to select the best features. So you'll notice in the fit algorithm below, the only modified part is inside the data_train[XXX]
dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring='accuracy', cv = cv_split)
dtree_rfe.fit(data_train[data_train_xled], data_train[Target].values.ravel())

X_rfe = data_train[data_train_xled].columns.values[dtree_rfe.get_support()]
rfe_results = model_selection.cross_validate(dtree, data_train[data_train_xled], data_train[Target].values.ravel(), cv=cv_split, return_train_score=True)

print('DT train shape after RFE: ', data_train.shape)
print('DT train columns after RFE: ', X_rfe)
print('DT train score mean after RFE: {}'.format(rfe_results['train_score'].mean()*100))
print('DT test score mean after RFE: {}'.format(rfe_results['test_score'].mean()*100))
print('DT test score std after REF: +/- {}'.format(rfe_results['test_score'].std()*100*3))
print('-'*10)

# Tune RFE model
param_grid = {'criterion':['gini', 'entropy'], 'max_depth':[2, 4, 6, 8, 10, None], 'random_state':[0], 'max_features':[3], 'min_samples_leaf':[1, 3, 10]}
rfe_tune_model = model_selection.GridSearchCV(dtree, param_grid=param_grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)
rfe_tune_model.fit(data_train[X_rfe], data_train[Target].values.ravel())

print('DT parameters after RFE tuned model: ', rfe_tune_model.best_params_)
print('DT train score after RFE tuned model: {}'.format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('DT test score mean after RFE tuned model: {}'.format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('DT test score std: +/- {}'.format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('-'*10)

print('\nIt looks like DecisionTree with hyperparameters gives 87.6% accuracy test with +/- 4.6 Std whereas DecisionTree with hyperparameter+RFE gives a similar test accuracy at 87.3% but higher std at +/- 7.4.\n')

# Decision Tree graphs

# import graphviz
# dt_graph = tree.export_graphviz(dtree, out_file=None, feature_names=data_train_xled, class_names=True, filled=True, rounded=True)
# graph = graphviz.Source(dt_graph)
# graph

# Heatmap for all algorithms/classifier prediction
# correlation_heatmap(MLA_predict)

# ------------ Random Forest with or without hyperparameters -------------
# without hyperparamters, Random Forest algorithm has been already tested beforehand. But here we will try it again just to be clear with a single algorithm performance
rdft = ensemble.RandomForestClassifier(random_state=0)
rdft_results = model_selection.cross_validate(rdft, data_train[data_train_xled], data_train[Target].values.ravel(), cv = cv_split, return_train_score=True)
rdft.fit(data_train[data_train_xled], data_train[Target].values.ravel())

print('RF parameters used: ', rdft.get_params())
print('RF train score mean: {}'.format(rdft_results['train_score'].mean()*100))
print('RF test score mean: {}'.format(rdft_results['test_score'].mean()*100))
print('RF test score std: {}'.format(rdft_results['test_score'].std()*100*3))
print('-'*10)

# setting up hyperparameters for Random Forest algorithms
cv_split = model_selection.ShuffleSplit(n_splits=10, train_size = 0.6, test_size = 0.3, random_state = 0)

param_grid = {'criterion':['gini', 'entropy'], 'max_depth':[2, 4, 6, 8, 10, None], 'random_state':[20], 'max_features':[5, 6, 7, 8], 'min_samples_leaf':[1, 2, 3, 7, 10]}
tune_model = model_selection.GridSearchCV(rdft, param_grid=param_grid, scoring='roc_auc', cv=cv_split, return_train_score=True)
tune_model.fit(data_train[data_train_xled], data_train[Target].values.ravel())

predRF = tune_model.predict(data_test[data_train_xled])

submission = pd.DataFrame({'PassengerId': data_test_id, 'Survived': predRF})
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('titanic_result_RFdummy.csv', index=False)

print('RF hyperparameters used: ', tune_model.best_params_)
print('RF train score mean with hyperparameters: {}'.format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('RF test score mean with hyperparameters: {}'.format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('RF test score std with hyperparameters: +/- {}'.format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('-'*10)

# ------------ Gradient Boosting with or without hyperparameters ----------
# without hyperparameters, GradientBoosting
gbc = ensemble.GradientBoostingClassifier(random_state=2)
gbc_results = model_selection.cross_validate(gbc, data_train[data_train_xled], data_train[Target].values.ravel(), cv = cv_split, return_train_score=True)
gbc.fit(data_train[data_train_xled], data_train[Target].values.ravel())

print('GB parameters used: ', gbc.get_params())
print('GB train score mean: {}'.format(gbc_results['train_score'].mean()*100))
print('GB test score mean: {}'.format(gbc_results['test_score'].mean()*100))
print('GB test score std: +/- {}'.format(gbc_results['test_score'].std()*100*3))
print('-'*10)

# setting up hyperparameters for Random Forest algorithms
param_grid = {'criterion':['friedman_mse'], 'learning_rate': [0.1], 'n_estimators':[100], 'max_depth':[2, 4, 6, 8, 10, None], 'random_state':[0], 'max_features':[5], 'min_samples_leaf':[1, 3, 10]}
gbc_tune_model = model_selection.GridSearchCV(gbc, param_grid=param_grid, scoring='roc_auc', cv=cv_split, return_train_score=True)
gbc_tune_model.fit(data_train[data_train_xled], data_train[Target].values.ravel())

print('GB hyperparameters used: ', gbc_tune_model.best_params_)
print('GB train score mean with hyperparameters: {}'.format(gbc_tune_model.cv_results_['mean_train_score'][gbc_tune_model.best_index_]*100))
print('GB test score mean with hyperparameters: {}'.format(gbc_tune_model.cv_results_['mean_test_score'][gbc_tune_model.best_index_]*100))
print('GB test score std with hyperparameters: +/- {}'.format(gbc_tune_model.cv_results_['std_test_score'][gbc_tune_model.best_index_]*100*3))
print('-'*10)


# ------------------- Voting Classifier ---------------------
# Voting classifier, temporarily disabled to run by coding with print """
vote_est = [
        ('ada', ensemble.AdaBoostClassifier()),
        ('bc', ensemble.BaggingClassifier()),
        ('etc', ensemble.ExtraTreesClassifier()),
        ('gbc', ensemble.GradientBoostingClassifier(criterion='friedman_mse', max_depth=10, random_state=0, max_features=8, min_samples_leaf=5)),
        ('rfc', ensemble.RandomForestClassifier(criterion='gini', max_depth=10, random_state=0, max_features=8, min_samples_leaf=5)),
        
        ('gpc', gaussian_process.GaussianProcessClassifier()),
        
        ('lr', linear_model.LogisticRegressionCV()),
        
        #('bnb', naive_bayes.BernoulliNB()),
        #('gnb', naive_bayes.GaussianNB()), 
        
        #('knn', neighbors.KNeighborsClassifier()),
        
        #('svc', svm.SVC(probability=True)),
        
        ('dtc', tree.DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=0, max_features=8, min_samples_leaf=5))
        
        ] 

# Use data_train[Target].values.ravel in cross_validate and fit because there's a false positive warning if you don't.

vote_hard = ensemble.VotingClassifier(estimators=vote_est, voting = 'hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, data_train[data_train_xled], data_train[Target].values.ravel(), cv=cv_split, return_train_score=True)
vote_hard.fit(data_train[data_train_xled], data_train[Target].values.ravel())
hardvote_pred = vote_hard.predict(data_test[data_train_xled])

submission = pd.DataFrame({'PassengerId': data_test_id, 'Survived': hardvote_pred})
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('titanic_result_hardvote.csv', index=False)

print('Hardvote train score mean: {}'.format(vote_hard_cv['train_score'].mean()*100))
print('Hardvote test score mean: {}'.format(vote_hard_cv['test_score'].mean()*100))
print('Hardvote test score std: +/- {}'.format(vote_hard_cv['test_score'].std()*100*3))
print('-'*10)

vote_soft = ensemble.VotingClassifier(estimators=vote_est, voting = 'soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, data_train[data_train_xled], data_train[Target].values.ravel(), cv=cv_split, return_train_score=True)
vote_soft.fit(data_train[data_train_xled], data_train[Target].values.ravel())
softvote_pred = vote_soft.predict(data_test[data_train_xled])

submission = pd.DataFrame({'PassengerId': data_test_id, 'Survived': softvote_pred})
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('titanic_result_softvote.csv', index=False)

print('Softvote train score mean: {}'.format(vote_soft_cv['train_score'].mean()*100))
print('Softvote test score mean: {}'.format(vote_soft_cv['test_score'].mean()*100))
print('Softvote test score std: +/- {}'.format(vote_soft_cv['test_score'].std()*100*3))
print('-'*10)


# -------------- PREDICTION ------------------
# prediction based on Decision Trees
# setting up hyperparameters for DT algorithms
# disable
print("""
param_grid = {'criterion':['gini', 'entropy'], 'max_depth':[2, 4, 6, 8, 10, None], 'random_state':[0], 'max_features':[5], 'min_samples_leaf':[1, 3, 10]}
tune_model = model_selection.GridSearchCV(dtree, param_grid=param_grid, scoring='roc_auc', cv=cv_split, return_train_score=True)
tune_model.fit(data_train[data_train_xled], data_train[Target])
pred = tune_model.predict(data_test[data_train_xled])

submission = pd.DataFrame({'PassengerId': data_test_id, 'Survived': pred})
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('titanic_result_DT.csv', index=False)

# prediction based on RandomForest
cv_split = model_selection.ShuffleSplit(n_splits=10, train_size = 0.8, test_size = 0.15, random_state = 0)

param_grid = {'criterion':['gini', 'entropy'], 'max_depth':[2, 4, 6, 7, 8, 9, 10, None], 'random_state':[20], 'max_features':[5, 7, 8], 'min_samples_leaf':[1, 2, 3, 7, 10]}
tune_model = model_selection.GridSearchCV(rdft, param_grid=param_grid, scoring='roc_auc', cv=cv_split, return_train_score=True)
tune_model.fit(data_train[data_train_xled], data_train[Target].values.ravel())

print('RF hyperparameters used: ', tune_model.best_params_)
print('RF train score mean with hyperparameters: {}'.format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('RF test score mean with hyperparameters: {}'.format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('RF test score std with hyperparameters: +/- {}'.format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('-'*10)

predRF = tune_model.predict(data_test[data_train_xled])

submission = pd.DataFrame({'PassengerId': data_test_id, 'Survived': predRF})
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('titanic_result_RF.csv', index=False)

# prediction based on Gradient Boosting
param_grid = {'criterion':['friedman_mse'], 'learning_rate': [0.1], 'n_estimators':[100], 'max_depth':[2, 4, 6, 8, 10, None], 'random_state':[0], 'max_features':[5], 'min_samples_leaf':[1, 3, 10]}
gbc_tune_model = model_selection.GridSearchCV(gbc, param_grid=param_grid, scoring='roc_auc', cv=cv_split, return_train_score=True)
gbc_tune_model.fit(data_train[data_train_xled], data_train[Target].values.ravel())
predGB = gbc_tune_model.predict(data_test[data_train_xled])

submission = pd.DataFrame({'PassengerId': data_test_id, 'Survived': predGB})
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('titanic_result_GB.csv', index=False)


# --------------- Testing -----------------
# disable

cv_split = model_selection.ShuffleSplit(n_splits=10, train_size = 0.7, test_size = 0.2, random_state = 0)
train_x, test_x, train_y, test_y = model_selection.train_test_split(data_train[data_train_xcalc], data_train[Target], random_state = 0)


train_xdummy, test_xdummy, train_ydummy, test_ydummy = model_selection.train_test_split(data_train_dummy[data_train_x_dummy], data_train[Target].values.ravel(), train_size = 0.8, test_size = 0.2, random_state = 20)
rdft = ensemble.RandomForestClassifier(criterion='gini', max_depth=20, random_state=0, max_features=16, min_samples_leaf=5)
rdft.fit(train_xdummy, train_ydummy) 
rdft.score(test_xdummy, test_ydummy)

train_xled, test_xled, train_yled, test_yled = model_selection.train_test_split(data_train[data_train_xled], data_train[Target].values.ravel(), train_size = 0.8, test_size = 0.2, random_state = 10)
rdft = ensemble.RandomForestClassifier(criterion='gini', max_depth=10, random_state=0, max_features=8, min_samples_leaf=5)
rdft.fit(train_xled, train_yled) 
rdft.score(test_xled, test_yled)

predRF = rdft.predict(data_test[data_train_xled])

submission = pd.DataFrame({'PassengerId': data_test_id, 'Survived': predRF})
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('titanic_result_RF.csv', index=False)

# gradient boosting
train_xdummy, test_xdummy, train_ydummy, test_ydummy = model_selection.train_test_split(data_train_dummy[data_train_x_dummy], data_train[Target].values.ravel(), train_size = 0.8, test_size = 0.2, random_state = 20)
gbcclf = ensemble.GradientBoostingClassifier(criterion='friedman_mse', max_depth=10, random_state=10, max_features=None, min_samples_leaf=5)
gbcclf.fit(train_xdummy, train_ydummy) 
gbcclf.score(test_xdummy, test_ydummy)

train_xled, test_xled, train_yled, test_yled = model_selection.train_test_split(data_train[data_train_xled], data_train[Target].values.ravel(), train_size = 0.8, test_size = 0.2, random_state = 10)
gbcclf.fit(train_xled, train_yled) 
gbcclf.score(test_xled, test_yled)


""")

# ---------------- Feature Importance testing from website-----------------
print("""
features = pd.DataFrame()
features['feature'] = data_train_xled
clf = ensemble.RandomForestClassifier()
clf = clf.fit(data_train[data_train_xled], data_train[Target].values.ravel())
features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(8,8))
""")





