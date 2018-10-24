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