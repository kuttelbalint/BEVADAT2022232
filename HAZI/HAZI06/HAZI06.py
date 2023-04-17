import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# load data
df = pd.read_csv("HAZI/HAZI06/NJ_60k.csv")
le = LabelEncoder()
df['status'] = le.fit_transform(df['status'])
df['line'] = le.fit_transform(df['line'])
df['type'] = le.fit_transform(df['type'])
df['day'] = le.fit_transform(df['day'])
df['part_of_the_day'] = le.fit_transform(df['part_of_the_day'])
# create feature and target variables
X = df.drop(['delay'], axis=1)
y = df['delay']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create decision tree classifier
dt_classifier = DecisionTreeClassifier()

# define hyperparameters grid
params = {'criterion': ['gini', 'entropy'],
'max_depth': [None, 2, 4, 6, 8, 10, 12],
'min_samples_split': [2, 3, 5, 8, 10, 20, 30],
'min_samples_leaf': [1, 2, 4, 8, 10, 20, 30]}

# perform grid search to find best hyperparameters
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=params, cv=5, verbose=1)
grid_search.fit(X_train, y_train)

# print best hyperparameters and corresponding accuracy score
print("Best hyperparameters: ", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print("Accuracy score: ", accuracy_score(y_test, y_pred))


"""
10 fitelés eredménye, és azok pontossága:
1.
    {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 8, 'min_samples_split': 2}
    Accuracy: 0.79775

2.
    {'criterion': 'entropy', 'max_depth': 8, 'min_samples_leaf': 12, 'min_samples_split': 2}
    Accuracy: 0.7915833333333333
3. 
     {'criterion': ['gini'],
          'max_depth': [4],
          'min_samples_split': [5],
          'min_samples_leaf': [1]}
    Accuracy: 0.7793333333333333

4.
    params = {'criterion': ['entropy'],
          'max_depth': [8],
          'min_samples_split': [10],
          'min_samples_leaf': [1]}
    Accuracy: 0.7915

5. 
    params = {'criterion': ['gini'],
          'max_depth': [2],
          'min_samples_split': [2],
          'min_samples_leaf': [1]}
    Accuracy: 0.7745

6. 
    params = {'criterion': ['entropy'],
          'max_depth': [6],
          'min_samples_split': [3],
          'min_samples_leaf': [2]}
    Accuracy: 0.7828333333333334

7.
    params = {'criterion': ['entropy'],
          'max_depth': [100],
          'min_samples_split': [20],
          'min_samples_leaf': [80]}
    Accuracy: 0.7983333333333333

"""