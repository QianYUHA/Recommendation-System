#Item-Based CF
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def run_itemcf():
    df=pd.read_csv("data/all.csv", dtype={'member_id': str, 'goods_id': str}, low_memory=False)

    user_counts = df['member_id'].value_counts()
    active_users = user_counts[user_counts >= 3].index
    df = df[df['member_id'].isin(active_users)]

    item_counts = df['goods_id'].value_counts()
    popular_items = item_counts[item_counts >= 5].index
    df = df[df['goods_id'].isin(popular_items)]
    #矩阵
    pivot=df.pivot_table(index='member_id', columns='goods_id', values='quanity', fill_value=0)

    #相似度
    item_sim=pd.DataFrame(cosine_similarity(pivot.T),index=pivot.columns,columns=pivot.columns)

    results=[]
    all_users = pivot.index
    top_k=10

    for user_id in all_users:
        user_vector = pivot.loc[user_id]
        interacted_items = user_vector[user_vector>0].index.tolist()

        scores = defaultdict(float)

        for item_i in interacted_items:
            similar_items=item_sim[item_i].drop(index=interacted_items)
            for item_j, sim_score in similar_items.items():
                scores[item_j]+=sim_score * user_vector[item_i]

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for item_id,score in sorted_scores:
            results.append({"member_id":user_id,"recommended_goods_id":item_id,"score":score})

    result_df=pd.DataFrame(results)
    result_df.to_csv("results/itemcf.csv", index=False)
    print("ItemCF 推荐结果已保存到 CSV！")

if __name__ == "__main__":
    run_itemcf()
# def recommend(user_id,top_k=5):
#    if user_id not in pivot.index:
#       print("用户不存在！")
#       return []

#    user_vector=pivot.loc[user_id]
#    user_interactions=user_vector[user_vector>0].index.tolist()

#    scores=pd.Series(dtype=float)

#    for item in user_interactions:
#        sim_items=item_sim[item]
#        scores=scores.add(sim_items,fill_value=0)

#    scores=scores.drop(labels=user_interactions,errors='ignore')
#    return scores.sort_values(ascending=False).head(top_k).index.tolist()

