
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_X_y

df = pd.read_csv(r"uber_clean.csv")
df = df[:15000]
scaler = MinMaxScaler()
scaler.fit(df[["Lat"]])
df['Lat'] = scaler.transform(df[["Lat"]])

scaler.fit(df[["Lon"]])
df["Lon"] = scaler.transform(df[["Lon"]])

kmeans_elkan = KMeans(n_clusters=9, algorithm="elkan")
labels_elkan = kmeans_elkan.fit_predict(df[["Lon", "Lat"]])
df["cluster"] = labels_elkan
print(kmeans_elkan.cluster_centers_)
print(labels_elkan)

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]
df6 = df[df.cluster == 5]
df7 = df[df.cluster == 6]
df8 = df[df.cluster == 7]
df9 = df[df.cluster == 8]

plt.scatter(df1.Lat, df1.Lon, color="green")
plt.scatter(df2.Lat, df2.Lon, color="red")
plt.scatter(df3.Lat, df3.Lon, color="black")
plt.scatter(df4.Lat, df4.Lon, color="yellow")
plt.scatter(df5.Lat, df5.Lon, color="blue")
plt.scatter(df6.Lat, df6.Lon, color="violet")
plt.scatter(df7.Lat, df7.Lon, color="indigo")
plt.scatter(df8.Lat, df8.Lon, color="orange")
plt.scatter(df9.Lat, df9.Lon, color="pink")

plt.scatter(kmeans_elkan.cluster_centers_[:, 1], kmeans_elkan.cluster_centers_[:, 0], color="purple", marker="*", label="centroid")

plt.xlabel("Lat")
plt.ylabel("Lon")
plt.legend()
plt.show()
#########################################################################
kmeans_lloyd = KMeans(n_clusters=9, algorithm="elkan")
labels_lloyd = kmeans_lloyd.fit_predict(df[["Lon", "Lat"]])
df["cluster"] = labels_lloyd
print(kmeans_lloyd.cluster_centers_)
print(labels_lloyd)

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]
df6 = df[df.cluster == 5]
df7 = df[df.cluster == 6]
df8 = df[df.cluster == 7]
df9 = df[df.cluster == 8]

plt.scatter(df1.Lat, df1.Lon, color="green")
plt.scatter(df2.Lat, df2.Lon, color="red")
plt.scatter(df3.Lat, df3.Lon, color="black")
plt.scatter(df4.Lat, df4.Lon, color="yellow")
plt.scatter(df5.Lat, df5.Lon, color="blue")
plt.scatter(df6.Lat, df6.Lon, color="violet")
plt.scatter(df7.Lat, df7.Lon, color="indigo")
plt.scatter(df8.Lat, df8.Lon, color="orange")
plt.scatter(df9.Lat, df9.Lon, color="pink")

plt.scatter(kmeans_lloyd.cluster_centers_[:, 1], kmeans_lloyd.cluster_centers_[:, 0], color="purple", marker="*", label="centroid")

plt.xlabel("Lat")
plt.ylabel("Lon")
plt.legend()
plt.show()
#########################################################################
kmeans_macqueen = KMeans(n_clusters=9, algorithm="macqueen")
labels_macqueen= kmeans_macqueen.fit_predict(df[["Lon", "Lat"]])
df["cluster"] = labels_macqueen
print(kmeans_macqueen.cluster_centers_)
print(labels_macqueen)

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]
df6 = df[df.cluster == 5]
df7 = df[df.cluster == 6]
df8 = df[df.cluster == 7]
df9 = df[df.cluster == 8]

plt.scatter(df1.Lat, df1.Lon, color="green")
plt.scatter(df2.Lat, df2.Lon, color="red")
plt.scatter(df3.Lat, df3.Lon, color="black")
plt.scatter(df4.Lat, df4.Lon, color="yellow")
plt.scatter(df5.Lat, df5.Lon, color="blue")
plt.scatter(df6.Lat, df6.Lon, color="violet")
plt.scatter(df7.Lat, df7.Lon, color="indigo")
plt.scatter(df8.Lat, df8.Lon, color="orange")
plt.scatter(df9.Lat, df9.Lon, color="pink")

plt.scatter(kmeans_macqueen.cluster_centers_[:, 1], kmeans_macqueen.cluster_centers_[:, 0], color="purple", marker="*", label="centroid")

plt.xlabel("Lat")
plt.ylabel("Lon")
plt.legend()
plt.show()


silhouette_elkan = silhouette_score(df[["Lat", "Lon"]], labels_elkan)
silhouette_lloyd = silhouette_score(df[["Lat", "Lon"]], labels_lloyd)
silhouette_macqueen = silhouette_score(df[["Lat", "Lon"]], labels_macqueen)

plt.bar(["Elkan's Algorithm", "Lloyd's Algorithm", "Macqueen's Algorithm"], [silhouette_elkan, silhouette_lloyd, silhouette_macqueen])
plt.xlabel("Clustering Algorithm")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Comparison")
plt.show()

# Elbow Method
# inertia = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, algorithm="lloyd")
#     kmeans.fit(df[["Lon", "Lat"]])
#     inertia.append(kmeans.inertia_)

# plt.plot(range(1, 11), inertia)
# plt.xlabel("Number of Clusters")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# plt.show()


# kmeans_elkan = KMeans(n_clusters=7, algorithm="elkan")
# labels_elkan = kmeans_elkan.fit_predict(df[["Lon", "Lat"]])
# # df["cluster"] = y_predicted
# print(kmeans_elkan.cluster_centers_)
# # print(y_predicted)

# kmeans_macqueen = KMeans(n_clusters=7, algorithm="macqueens")
# labels_macqueen = kmeans_macqueen.fit_predict(df[["Lon", "Lat"]])
# # df["cluster"] = y_predicted
# print(kmeans_macqueen.cluster_centers_)

# silhouette_lloyd = silhouette_score(df[["Lon", "Lat"]], labels_lloyd)
# silhouette_elkan = silhouette_score(df[["Lon", "Lat"]], labels_elkan)
# silhouette_macqueen = silhouette_score(df[["Lon", "Lat"]], labels_macqueen)

# print(f"Silhouette Score - MacQueen's Algorithm: {silhouette_macqueen}")
# print(f"Silhouette Score - Lloyd's Algorithm (KMeans): {silhouette_lloyd}")
# print(f"Silhouette Score - Elkan's Algorithm: {silhouette_elkan}")


