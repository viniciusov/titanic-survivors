# coding: utf-8

# ## Importing Data

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

main_df = pd.read_csv("train.csv")
final_df = pd.read_csv("test.csv")
both_df = pd.concat([main_df, final_df])

names = {}
counter = 1

for item in both_df['Name']:
    if item.split(",")[1].split(".")[0].strip(" ") not in names:
        names[item.split(",")[1].split(".")[0].strip(" ")] = counter
        counter += 1      

age_mean = both_df["Age"].mean() # Get age from both datasets

# Get average data from name title (don't know why, but the score with this is worse, so I disable it)

age_by_name = {}
counter = 0
age = 0

for name in names:
    for n,item in enumerate(both_df['Name']):
        if item.split(",")[1].split(".")[0].strip(" ") == name:
            if pd.notna(both_df.iloc[n]['Age']):
                counter += 1
                age += both_df.iloc[n]['Age']
    age_by_name[name] = age/counter # Get the average
    counter = 0
    age = 0

deck_dict = {}
counter = 1

for item in set(both_df['Cabin']):
    if not(pd.isna(item)) and item[0] not in deck_dict:
        deck_dict[item[0]]=counter
        counter += 1

cabin_average = {}
counter = 0
fare = 0

for deck in deck_dict:
    for n,item in enumerate(both_df['Cabin']):
        if pd.notna(item) and item[0] == deck:
            counter += 1
            fare += both_df.iloc[n]['Fare']    
    cabin_average[fare/counter] = deck
    counter = 0
    fare = 0

# Testing this function to get the nearest value from a list (using '18' only for testing)
# For every item on that list, lambda function will use 'abs(item-18)' instead of 'item', the choose the lower 'item'
cabin_average[min(list(cabin_average.keys()), key=lambda x:abs(x-18))] 

# ## Preparing Data

def preprocess(df):
    df = df.drop(["PassengerId","Ticket"], axis=1)
    
    for n,item in enumerate(df["Cabin"]):
        if pd.isna(item):
            cabin_key = cabin_average[min(list(cabin_average.keys()), key=lambda x:abs(x-df["Fare"][n]))]
            df.at[n,"Cabin"] = deck_dict[cabin_key]
        else:
            df.at[n,"Cabin"] = deck_dict[item[0]]

    for n,item in enumerate(df["Age"]):
        if pd.isna(item):
            #df.at[n,"Age"] = age_by_name[df.iloc[n]["Name"].split(",")[1].split(".")[0].strip(" ")]
            df.at[n,"Age"] = age_mean
    
    for n,item in enumerate(df["Name"]):
        if item.split(",")[1].split(".")[0].strip(" ") in names:
            df.at[n,"Name"] = names[item.split(",")[1].split(".")[0].strip(" ")]
        else:
            df.at[n,"Name"] = 0
    
    for n,item in enumerate(df["Sex"]):
        if item == "male":
            df.at[n,"Sex"] = 0 #According documentation, use dt.at[row, 'collumn'] instead df['column'][row]
        elif item == "female":
            df.at[n,"Sex"] = 1
        else:
            df.at[n,"Sex"] = 2 #Just in case
            
    for n,item in enumerate(df["Embarked"]):
        if item == "S":
            df.at[n,"Embarked"] = 0
        elif item == "C":
            df.at[n,"Embarked"] = 1
        else:
            df.at[n,"Embarked"] = 2 #Fill 'Q' or missing values with 2
            
    df = df.fillna(0)
            
    return df

# ## Train/Test Split

train_df = preprocess(main_df)
train_df_feat = train_df.iloc[:,1:]
train_df_class = train_df.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(train_df_feat, train_df_class, test_size=0.10, random_state=101)

# ## Balancing Data (Over-sampling)

total = len(y_train)
survived = list(y_train).count(1)
        
survived_pct = survived/total

sm = SMOTE(random_state=101)
X_train, y_train = sm.fit_resample(X_train, y_train)

total = len(y_train)
survived = list(y_train).count(1)
        
survived_pct = survived/total
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# ## Building Random Forest

random_forest = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=101)
random_forest.fit(scaled_X_train, y_train)
random_forest.score(scaled_X_train, y_train)

# ## Testing

prediction = random_forest.predict(scaled_X_test)
random_forest.score(scaled_X_test, y_test)

# Another Metric
print("oob score:", round(random_forest.oob_score_, 4)*100, "%")

# ## Hyperparameters tuning

# A manually way to pick-up the hyperparameters for the best test scores

for critery in ["gini", "entropy"]:
    for min_leaf in [1, 3, 4,5]:
        for min_split in [2, 3, 10]:
            for estimator in [50, 100, 200, 300, 1000]:

                rf2 = RandomForestClassifier(criterion=critery,
                                             min_samples_leaf=min_leaf,
                                             min_samples_split=min_split,
                                             n_estimators=estimator,
                                             oob_score=True, 
                                             random_state=101) # creates a new estimator
                rf2.fit(scaled_X_train, y_train)

                print('critery:', critery,
                      'min_leaf:', min_leaf,
                      'min_split:', min_split,
                      'estimator:', estimator)
                print('Train Score:',rf2.score(scaled_X_train, y_train))
                print('Test Score:',rf2.score(scaled_X_test, y_test))


# ###### *As a suggestion, you can use the AdaBoostClassifier or GradientBoostingClassifier, but the results were worse for this problem.*

# # Best result => 'critery:  entropy  max_feature:  3  min_leaf:  4  min_sample:  10  estimator:  100'
# rf_final = RandomForestClassifier(criterion='entropy', min_samples_leaf=4, min_samples_split=10, 
#                                   n_estimators=100, oob_score=True, random_state=101)
# rf_final.fit(scaled_X_train, y_train)
# rf_final.score(scaled_X_test, y_test)


# ## Checking importances

train_df.columns

random_forest.feature_importances_

importances = pd.DataFrame({'feature':train_df_feat.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)

get_ipython().run_line_magic('matplotlib', 'inline')

importances.plot.bar()


# ## Final Prediction

submission_df = preprocess(final_df)
submission_df.head(10)

scaled_X_final = scaler.transform(submission_df)
scaled_X_final

final_prediction = random_forest.predict(scaled_X_final)
final_prediction

final = pd.concat([final_df['PassengerId'],pd.DataFrame(final_prediction, columns=['Survived'])], axis=1)
final.head(10)

with open('submission.csv','w') as csv_file:
    csv_file.write('PassengerId,Survived\n')
    
    for index,row in final.iterrows():
        text = str(row['PassengerId'])+','+str(row['Survived'])
        csv_file.write(text+'\n')

