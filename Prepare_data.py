import pandas as pd

df1=pd.read_csv('data/data_114_20241107.txt', sep='|', encoding='utf-8-sig',dtype={'member_id': str, 'phone': str})
df2=pd.read_csv('data/data_itv_20250311.txt', sep='|', encoding='utf-8-sig',dtype={'member_id': str, 'phone': str})
df3=pd.read_csv('data/data_qyi_20250311.txt', sep='|', encoding='utf-8-sig',dtype={'member_id': str, 'phone': str})
df4=pd.read_csv('data/data_yup_20250311.txt', sep='|', encoding='utf-8-sig',dtype={'member_id': str, 'phone': str})

df_all=pd.concat([df1,df2,df3,df4],ignore_index=True)


df_all=df_all.dropna(subset=['member_id','goods_id'])
df_all=df_all.drop_duplicates()

df_rec=df_all[['member_id','goods_id','is_order','quanity','business']]

df_rec.to_csv('data/all.csv',index=False)

df=pd.read_csv('data/all.csv',dtype={'member_id':str,'goods_id':str})
df['label']=(df['quanity']>0).astype(int) #用历史行为生成正样本标签
df.to_csv("data/all_with_label.csv",index=False)
