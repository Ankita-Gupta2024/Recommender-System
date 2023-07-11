import pandas as pd
import numpy as np
import time
import fastparquet as fp
from scipy import stats
from lightfm import LightFM
from scipy.sparse import csr_matrix
from lightfm.cross_validation import random_train_test_split
from sklearn.model_selection import KFold, RandomizedSearchCV
from lightfm.evaluation import precision_at_k, auc_score, auc_score,recall_at_k,reciprocal_rank
# pip install lightfm -- to execute lightFM

#setting the base path for large dataset
base_path = '/scratch/work/courses/DSGA1004-2021/listenbrainz/'
print('Training Dataset input')
train_data=pd.read_parquet('{}/interactions_train_small.parquet'.format(base_path), engine='fastparquet')
train_data=train_data.reindex(columns=['user_id', 'recording_msid', 'timestamp'])

print('Testing Dataset input')
test_data=pd.read_parquet('{}/interactions_test.parquet'.format(base_path), engine='fastparquet')
test_data=test_data.reindex(columns=['user_id', 'recording_msid', 'timestamp'])

#appending the train and test data
complete_data = pd.concat([train_data, test_data], ignore_index=True)

num_users = complete_data['user_id'].nunique()
num_items = complete_data['recording_msid'].nunique()
print('Number of unique users: {}, Number of unique recording IDs {} in train.'.format(num_users, num_items))

#calculating playcount column similar to ALS
complete_data = complete_data.groupby(["user_id", "recording_msid"]).size().reset_index(name="play_count")

row_count = complete_data.shape[0]
print("Count of rows in train_data:", row_count)

# Find the maximum value of the 'play_count' column 
max_play_count = complete_data['play_count'].max()

# Retrieve the rows with the maximum play_count value
max_play_count_rows = complete_data[complete_data['play_count'] == max_play_count]

# Print the rows with the maximum play_count value
print("Rows with maximum play_count:")
print(max_play_count_rows)

#setting the threshold as 30 (same as ALS)
filtered_train_data = complete_data[complete_data['play_count'] > 30]

# Get the count of rows in the filtered DataFrame
row_count = filtered_train_data.shape[0]
print("Count of rows in train_data where 'play_count' > 30:", row_count)

#creating stringIndexer in LightFM
mapping_dict = {}
index = 0
for value in filtered_train_data['recording_msid'].unique():
    mapping_dict[value] = index
    index += 1

#adding the indexing column
filtered_train_data['recording_index'] = filtered_train_data['recording_msid'].map(mapping_dict)

#dropping recording
filtered_train_data = filtered_train_data.drop('recording_msid', axis=1)

# Creating a sparse matrix for play count along with user id and recording msid
interactions_matrix = csr_matrix((filtered_train_data['play_count'],
                                  (filtered_train_data['user_id'], filtered_train_data['recording_index'])))

#splitting the data in training n validation 
(train_matrix, test_matrix) = random_train_test_split(interactions_matrix, test_percentage=0.2)

rank = 10
reg_param = 0.05
top = 100

#want to maximize precision at K so type of loss is warp
model = LightFM(loss='warp', no_components=rank, user_alpha=reg_param,item_alpha=reg_param random_state=1,learning_rate=0.02)

begin_time = time.time()
model = model.fit(train_matrix, epochs=20)
fit_time = (time.time() - begin_time)/60

#evaluation of test matrix
begin_evaluation_time = time.time()
precision_at_K = precision_at_k(model, test_matrix, k=top).mean()
AUC_score = auc_score(model, test_matrix).mean()
recall_at_K = recall_at_k(model, test_matrix, k=top).mean()
reci_rank = reciprocal_rank(model, test_matrix).mean()
total_evaluation_time = (time.time() - begin_evaluation_time)/60

#evaluation of train matrix
begin_evaluation_time2 = time.time()
precision_at_K_train = precision_at_k(model, train_matrix, k=top).mean()
AUC_score_train = auc_score(model, train_matrix).mean()
recall_at_K_train = recall_at_k(model, train_matrix, k=top).mean()
reci_rank_train = reciprocal_rank(model, train_matrix).mean()
total_evaluation_time2 = (time.time() - begin_evaluation_time2)/60


print('trained model with:')
print("reg_param = ", reg_param)
print("rank = ",rank)
print(f'Fit time on training data {fit_time} mins')
print(f'Precision at K: {precision_at_K_train}')
print(f'AUC score: {AUC_score_train}')
print(f'Recall at K: {recall_at_K_train}')
print(f'Reciprocal Rank: {reci_rank_train}')
print(f'Run time for evaluation {total_evaluation_time2} mins')
print('The values of testing model is:')
print(f'Precision at K: {precision_at_K}')
print(f'AUC score: {AUC_score}')
print(f'Recall at K: {recall_at_K}')
print(f'Reciprocal Rank: {reci_rank}')
print(f'Run time for evaluation {total_evaluation_time} mins')