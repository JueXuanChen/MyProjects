import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# 匯入資料
df = pd.read_csv('Financial Indicators_X1.csv', sep=',')
df.head()

# 將不必納入分群的資料(也就是公司名稱)給剃除
x = df.drop('Company Name', axis=1)
x.head()

# 使用 Elbow Method 尋找最佳的群集數量
WCSS = []

for i in range(1, 11):
  kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=1500, n_init=10, random_state=0)
  kmeans.fit(x)
  WCSS.append(kmeans.inertia_)

print(WCSS)

plt.plot(range(1, 11), WCSS)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('Elbow Method.png')
plt.show()

# 使用選定的群集數量(k=6)進行最終分群並視覺化結果
kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=1500, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(x)

plt.scatter(x.iloc[:, 0], x.iloc[:, 1],) # c=pred_y, cmap='rainbow'
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, color='red')
plt.savefig('KMeans.png')
plt.show()

# 保存群集中心和分群結果到 CSV 檔案
Centers = kmeans.cluster_centers_
Clustering_Centers = pd.DataFrame(Centers)
Clustering_Centers.to_csv('/財務指標_Clustering_Centers_KMeans.csv', mode='a', header=True)

Clustering_results = pd.DataFrame(df)
Clustering_results ['Clustering_results'] = pred_y

Clustering_results.to_csv('財務指標_Clustering_results_KMeans.csv', mode='a', header=True)

