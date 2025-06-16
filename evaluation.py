from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm


df = pd.read_csv("data/all.csv", dtype={'member_id': str, 'goods_id': str}, low_memory=False)

train, test = train_test_split(df, test_size=0.2, random_state=42)

def evaluate_itemcf(train_df, test_df, top_k=10):
    print(f"原始训练数据大小: {train_df.shape}")

    user_counts = train_df['member_id'].value_counts()
    active_users = user_counts[user_counts >= 2].index
    train_df = train_df[train_df['member_id'].isin(active_users)] #只保留活跃用户防止内存爆炸

    item_counts = train_df['goods_id'].value_counts()
    popular_items = item_counts[item_counts >= 3].index
    train_df = train_df[train_df['goods_id'].isin(popular_items)]

    print(f"过滤后训练数据大小: {train_df.shape}")

    if len(train_df) > 10000:
        train_df = train_df.sample(n=10000, random_state=42)
        print("对训练集进行了抽样，规模为10000")

    test_df = test_df[test_df['member_id'].isin(train_df['member_id'].unique())]
    pivot = train_df.pivot_table(index='member_id', columns='goods_id', values='quanity', fill_value=0)
    item_sim = pd.DataFrame(cosine_similarity(pivot.T), index=pivot.columns, columns=pivot.columns)

    hit = 0
    total_pred = 0
    total_true = 0

    for user_id in tqdm(test_df['member_id'].unique()):
        if user_id not in pivot.index:
            continue

        user_vector = pivot.loc[user_id]
        interacted_items = user_vector[user_vector > 0].index.tolist()
        if not interacted_items:
            continue

        test_items = test_df[test_df['member_id'] == user_id]['goods_id'].tolist()
        if len(test_items) == 0:
            continue

        scores = defaultdict(float)
        for item_i in interacted_items:
            similar_items = item_sim[item_i].drop(index=interacted_items, errors='ignore')
            for item_j, sim_score in similar_items.items():
                scores[item_j] += sim_score * user_vector[item_i]

        recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        recommended_items = [item for item, _ in recommended]

        hit += len(set(recommended_items) & set(test_items))
        total_pred += len(recommended_items)
        total_true += len(test_items)

    precision = hit / total_pred if total_pred > 0 else 0
    recall = hit / total_true if total_true > 0 else 0
    print(f"Precision@{top_k}: {precision:.4f}, Recall@{top_k}: {recall:.4f}")

evaluate_itemcf(train, test, top_k=10)

