# Mahdi Abdollahpour, 09/04/2022, 03:16 PM, PyCharm, lgeconvrec

import numpy as np
import pandas as pd
from Neural_PM.utils.exp import *
from Neural_PM.utils.eval import *
from Neural_PM.prefernce_matching.matrix import *
from Neural_PM.utils.load_data import *
from Neural_PM.prefernce_matching.PM import *
import os


queries, restaurants = get_queries_and_resturants('../data/CRS dataset-Restaurantwith sample reviews - Binary Data.csv')
restaurants = restaurants.tolist()

df = pd.read_csv("../data/73_restaurants_all_rates.csv")
# df = pd.read_csv("../data/73_restaurants_all_rates.csv")
print(df['name'])

df = df[df['name'].isin(restaurants)]
print(df.groupby('business_id').count())
df.to_csv("../data/50_restaurants_all_rates.csv")
# print(df.columns)



