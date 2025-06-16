import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm
import joblib

model = keras.models.load_model('models/twotower_model2.h5')
user_encoder = joblib.load("models/user_encoder.pkl")
item_encoder = joblib.load("models/item_encoder.pkl")

df = pd.read_csv("data/train_data.csv", dtype={"member_id": str, "goods_id": str})

#过滤未知用户
df = df[df['member_id'].isin(user_encoder.classes_)]
df = df[df['goods_id'].isin(item_encoder.classes_)]

df['user'] = user_encoder.transform(df['member_id'])
df['item'] = item_encoder.transform(df['goods_id'])

all_user_ids = df['user'].unique()
all_item_ids = df['item'].unique()

#反编码字典
inv_item_map = {i: id_ for id_, i in zip(df['goods_id'], df['item'])}
inv_user_map = {i: id_ for id_, i in zip(df['member_id'], df['user'])}

topK=10
recommendations = []

for user_id in tqdm(all_user_ids):
    user_tensor = np.full(len(all_item_ids), user_id)
    item_tensor = all_item_ids

    scores = model.predict([user_tensor, item_tensor], batch_size=1024,verbose=0).reshape(-1)
    top_indices = scores.argsort()[-topK:][::-1]
    top_items = item_tensor[top_indices]

    for item in top_items:
        recommendations.append((inv_user_map[user_id], inv_item_map[item]))

rec_df = pd.DataFrame(recommendations,columns=['user_id', 'item_id'])
rec_df.to_csv("recommendation2.csv",index=False)
print("推荐完成")