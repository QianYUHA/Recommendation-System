import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import  keras
from tensorflow.keras import layers, regularizers
import pandas as pd
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#train_data = pd.read_csv("data/train_data.csv", dtype={'member_id': str, 'goods_id': str})
#热门
#user_counts = train_data['member_id'].value_counts()
#item_counts = train_data['goods_id'].value_counts()
#active_users = user_counts[user_counts >= 5].index
#popular_items = item_counts[item_counts >= 5].index
#train_data = train_data[
#    (train_data['member_id'].isin(active_users)) &
#    (train_data['goods_id'].isin(popular_items))
#]
#train_data = train_data.sample(n=20000, random_state=42).reset_index(drop=True)

df = pd.read_csv("data/train_data.csv",dtype={"member_id": str, 'goods_id': str})
sample_users = df['member_id'].drop_duplicates().sample(n=50000, random_state=42)
df = df[df['member_id'].isin(sample_users)]

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df['user'] = user_encoder.fit_transform(df['member_id'])
df['item'] = item_encoder.fit_transform(df['goods_id'])

num_users = df['user'].nunique()
num_items = df['item'].nunique()

X = df[['user', 'item']]
y = df['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

embedding_dim = 64

#用户塔
user_input=keras.Input(shape=(1,),name='user')
user_embedding=layers.Embedding(num_users,embedding_dim,name='user_embedding')(user_input)
user_vector = layers.GlobalAveragePooling1D()(user_embedding)
user_vector = layers.Dense(64,activation='relu')(user_vector)
user_vector = layers.Dropout(0.3)(user_vector)
user_vector = layers.Dense(32,activation='relu')(user_vector)

#商品塔
item_input=keras.Input(shape=(1,),name='item')
item_embedding=layers.Embedding(num_items, embedding_dim,name='item_embedding')(item_input)
item_vector = layers.GlobalAveragePooling1D()(item_embedding)
item_vector = layers.Dense(64, activation='relu')(item_vector)
item_vector = layers.Dropout(0.3)(item_vector)
item_vector = layers.Dense(32, activation='relu')(item_vector)

#相似度打分
dot_product=layers.Dot(axes=1)([user_vector,item_vector])
output=layers.Activation('sigmoid')(dot_product)

model=keras.Model(inputs=[user_input,item_input],outputs=[output])
model.compile(optimizer=Adagrad(),loss='binary_crossentropy',metrics=['accuracy'])

#early stopping
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    ModelCheckpoint('models/best_twotower_model.h5', save_best_only=True, monitor='val_loss')
]


model.fit(
    [X_train['user'], X_train['item']], y_train,
    validation_data=([X_val['user'], X_val['item']], y_val),
    epochs=5,
    batch_size=256
)

model.save("models/twotower_model2.h5")

# 保存编码器
import joblib
joblib.dump(user_encoder, "models/user_encoder.pkl")
joblib.dump(item_encoder, "models/item_encoder.pkl")