import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('HAZI/HAZI06/NJ_60k.csv')
le = LabelEncoder()
data['status'] = le.fit_transform(data['status'])
data['line'] = le.fit_transform(data['line'])
data['type'] = le.fit_transform(data['type'])
data['day'] = le.fit_transform(data['day'])
data['part_of_the_day'] = le.fit_transform(data['part_of_the_day'])


X = data.drop(['delay'], axis=1)
y = data['delay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)


model = DecisionTreeClassifier( criterion='gini', max_depth=12, min_samples_leaf=10, random_state=41)
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
(f'Accuracy: {accuracy:.2%}')

"""
10 fitelés eredménye, és azok pontossága:
1. 
    criterion='entropy', max_depth=12, min_samples_leaf=10, random_state=41
    Accuracy: 80.19%
2.
    criterion='entropy', max_depth=18, min_samples_leaf=10, random_state=41
    Accuracy: 79.47%
3.
    criterion='entropy', max_depth=18, min_samples_leaf=5, random_state=41
    Accuracy: 78.66%
4.
    criterion='gini', max_depth=18, min_samples_leaf=5, random_state=41
    Accuracy: 78.75%
5.
    criterion='gini', max_depth=6, min_samples_leaf=12, random_state=41
    Accuracy: 78.83%
6.
    criterion='gini', max_depth=60, min_samples_leaf=12, random_state=41
    Accuracy: 79.24%
7.
    criterion='gini', max_depth=60, min_samples_leaf=50, random_state=41
    Accuracy: 79.96%
8.
    criterion='entropy', max_depth=60, min_samples_leaf=66, random_state=41
    Accuracy: 79.74%
9. 
    criterion='gini', max_depth=12, min_samples_leaf=10, random_state=41
    Accuracy: 80,37%
10.
    criterion='gini', max_depth=14, min_samples_leaf=66, random_state=41
    Accuracy: 79.66

Az általam talált legjobb paramérek: 9. teszteset


Összefoglalás:

    Számomra a legnagyobb kihívást a feladat megértése, valamint a decisionTree modell 
    értelmezése és implementálása jelentett. Elméletben már értettem, hogy hogyan működik, 
    de így ezen a "komplexebb" gyakorlati példán már nehezebbnek bizonyosult a feladat implementálása
    python nyelvben. Számos error message-t kaptam, melyek néha nem voltak egyértelműek - köztük a csv betöltésénél, az oszlopok típusánál.
    Végül a paraméterek megtalálásánál akadtam el hosszabb időre. Végül sikerült olyan paramétereket (random módón:D)
    találnom, melyek segítségével 80% fölé tudtam vinni az accuracy-t. A következő "célom" az lenne, hogy a paraméterezésnél
    legyen valamilyen céltudatosságom, jobban megértsem, hogy az egyes paraméterek változtatása milyen irányba viszi majd el
    a pontosságot.
"""

