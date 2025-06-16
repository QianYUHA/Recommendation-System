import pandas as pd
import numpy as np

df=pd.read_csv("data/all_with_label.csv",dtype={'member_id': str, 'goods_id': str})

positive=df[df['label']==1][['member_id', 'goods_id']]
positive['label']=1

all_users=df['member_id'].unique()
all_items=df['goods_id'].unique()

neg=[]
for user in np.random.choice(all_users,size=5000,replace=False):
    user_pos_item=set(positive[positive['member_id']==user]['goods_id'])
    for _ in range(5):
        item=np.random.choice(all_items)
        while item in user_pos_item:
            item=np.random.choice(all_items)
        neg.append([user,item,0])

neg_df=pd.DataFrame(neg,columns=['member_id','goods_id','label'])

#合并正负样本
train_data=pd.concat([positive,neg_df],ignore_index=True)
train_data=train_data.drop_duplicates()
train_data.to_csv("data/train_data.csv",index=False)
