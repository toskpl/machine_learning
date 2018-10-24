import pandas as pd
import numpy as np

#ustwiamy ziarnko (seed), dla powtarzalnosci w losowaniu
np.random.seed(2018) # <== moze byc dowolna liczba, ale trzeba wybrac cos... wiec rok 2018 :)

#models (algorithms)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#walidacja wyniku
from sklearn.model_selection import train_test_split

#metryka sukcesu
from sklearn.metrics import accuracy_score # <== our success metric

#wizualizacja
import matplotlib.pyplot as plt

# <== umozliwia robic wykresy w notebook'u, zamiast otwierac w oknie
%matplotlib inline

df = pd.read_csv("../input/polish_names.csv")

df['target'] = df['gender'].map(lambda x: int(x == 'm'))
df


def train_and_predict_model(X_train, X_test, y_train, y_test, model, success_metric=accuracy_score):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return success_metric(y_test, y_pred)

vowels = ['a', 'a', 'e', 'e', 'i', 'o', 'u', 'y']

def how_many_vowels(name):
    return sum( map(lambda x: int(x in vowels), name.lower()) )
    
def first_is_vowel(name):
    return name.lower()[0] in vowels

def last_is_vowel(name):
    return name.lower()[-1] in vowels

def get_all_vowels(name):
    all_vowels = [letter for letter in name.lower() if letter in vowels]
    
    return ''.join(all_vowels)

def get_all_consonants(name):
    all_consonants = [letter for letter in name.lower() if letter not in vowels]
    
    return ''.join(all_consonants)

def feature_engineering(df):
    df['len_name'] = df['name'].map(lambda x: len(x))
    
    df['first_is_vowel'] = df['name'].map(first_is_vowel)
    df['last_is_vowel'] = df['name'].map(last_is_vowel)
    
    df['first_letter'] = df['name'].map(lambda x: x.lower()[0])
    df['first_letter_cnt'] = pd.factorize(df['first_letter'])[0]
    
    df['last_letter'] = df['name'].map(lambda x: x.lower()[-1])
    df['last_letter_cnt'] = pd.factorize(df['last_letter'])[0]
    
    df['all_vowels'] = df['name'].map(get_all_vowels)
    df['all_vowels_cnt'] = pd.factorize(df['all_vowels'])[0]

    df['all_consonants'] = df['name'].map(get_all_consonants)
    df['all_consonants_cnt'] = pd.factorize(df['all_consonants'])[0]
    
    
    return df
	
features = ['len_name', 'first_is_vowel', 'last_is_vowel', 'first_letter_cnt', 'last_letter_cnt', 'all_vowels_cnt', 'all_consonants_cnt']
X = df_fe[ features ]
y = df_fe['target']

train_and_predict_model(X, X, y, y, LogisticRegression())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)
y_train
print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)