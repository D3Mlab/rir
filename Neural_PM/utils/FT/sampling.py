from itertools import combinations
import pandas as pd
import numpy as np
import pickle


def get_positve_samples(address_of_data = '/content/400_review_3_star_above - 400_review_3_star_above.csv',address_to_save ='positive_sample.pkl' ):
  '''
  Generate all the combinations of reviews per restaurant
  return: dataframe with original data and review ID, return combination of restaurant in a csv to reviewid1,rating1,reviewid2,rating2,business_id
  '''
  all_reviews = pd.read_csv(address_of_data)
  all_reviews['review_id'] = all_reviews['business_id']+all_reviews['user_id']
  restaurants = all_reviews.business_id.unique()
  all_positive_samples = {'review_id1': [],
                          'rating1': [],
                          'review_id2': [],
                          'rating2': [],
                          'business_id': []
                          }
  all_len = 0
  for restaurant in restaurants:
    temp_df = all_reviews[all_reviews['business_id']==restaurant][['review_id','review_stars']]
    print(temp_df.shape,restaurant)
    id_rating_tuple = [tuple(x) for x in temp_df.to_numpy()]
    positive_samples = list(combinations(id_rating_tuple, 2))
    all_len += len(positive_samples)
    all_positive_samples[restaurant] = positive_samples
  print('number of all positive samples:' , all_len)
  for sample in positive_samples:
      all_positive_samples['business_id'].append(restaurant)
      all_positive_samples['review_id1'].append(sample[0][0])
      all_positive_samples['rating1'].append(sample[0][1])
      all_positive_samples['review_id2'].append(sample[1][0])
      all_positive_samples['rating2'].append(sample[1][1])
  compression_opts = dict(method='zip',
                          archive_name='all_samples.csv')
  all_positive_samples.to_csv('all_samples.zip', index=False,
                              compression=compression_opts)
  return all_positive_samples,all_reviews