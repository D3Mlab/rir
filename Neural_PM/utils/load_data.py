# Mahdi Abdollahpour, 16/03/2022, 10:51 AM, PyCharm, Neural_PM

import pandas as pd
import json
def get_queries_and_resturants(
        path_of_labels='./data/CRS dataset-Restaurantwith sample reviews - Binary Data.csv'):
    df = pd.read_csv(path_of_labels)
    queries = df["query"].unique()
    restaurants = df["Restaurant name"].unique()
    return queries, restaurants


def sort_reviews_by_business(review_df, restaurants, col_name="review_text"):
    reviews = {}
    id_name_map = {}
    already_seen = []
    for index, row in review_df.iterrows():
        if row["name"] in restaurants:

            id_name_map[row["business_id"]] = row["name"]

            if row["business_id"] in reviews.keys():
                reviews[row["business_id"]] += [row[col_name]]
            else:
                if row["name"] in already_seen:
                    print(row["name"])
                already_seen.append(row["name"])
                reviews[row["business_id"]] = [row[col_name]]
    # print(already_seen)
    return reviews, id_name_map



def load_toronto_dataset(filename='/content/drive/MyDrive/CRS_DATA/Cleaned_Toronto_Reviews.json', sample=1):
    with open(filename, 'r') as f:
        data = f.readlines()
        data = list(map(json.loads, data))

    data = data[0]

    df = pd.DataFrame(data)
    df.rename(columns={'stars': 'review_stars', 'text': 'review_text'},
              inplace=True)

    df = df[['business_id', 'user_id', 'review_stars', 'review_text', 'name', 'categories', 'date']]
    df=df[df['review_text']!="This review has been removed for violating our Terms of Service"]
    df = df[df['review_stars']>2.5]
    if(not isinstance(sample, int)):
        df=df.sample(frac=sample)
    elif (sample!=1):
        df=df.sample(n=sample)

    df.set_index(['business_id', 'user_id', 'review_stars', 'review_text', 'name', 'categories', 'date'])
    return df