import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pandas import DataFrame
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


img = cv2.imread(
    "/home/keerthivasan/.ros/image2.png", cv2.IMREAD_GRAYSCALE)

blank_img = np.zeros(
    (img.shape[0], img.shape[1], 3), dtype=np.uint8)
print(f"{img.shape[0]}\t{img.shape[1]}")


def poly_value(a, b, c, xy):
    global blank_img
    global img
    for element in xy[:, 0]:
        img = cv2.circle(img, tuple(
            [int(a*element*element+b*element+c), int(element)]), 0, (255, 0, 255), 5)
        print(tuple([int(element), abs(int(a*element*element+b*element+c))]))


#print(img.shape[1], img.shape[0])
indexes_points = []
for index, element in np.ndenumerate(img):
    if element == 255:
        indexes_points.append([index[0], index[1]])

# note the first column represents the y values
# the second column represents the x values

indexes_points = np.array(indexes_points)
# print(len(indexes_points))
# rint(type(indexes_points))
# print(np.sort(indexes_points))

#df = DataFrame(dict(x=indexes_points[:, 1], y=indexes_points[:, 0]))
#fig, ax = plt.subplots(figsize=(8, 8))
#df.plot(ax=ax, kind='scatter', x='x', y='y')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

'''
neighbors = NearestNeighbors(n_neighbors=10, radius=4)
neighbors_fit = neighbors.fit(indexes_points)
distances, indices = neighbors_fit.kneighbors(indexes_points)
print(indices)
distances = np.sort(distances, axis=1)
distances = distances[:, 1]

plt.plot(distances)
plt.show()
'''
X = indexes_points
db = DBSCAN(eps=25, min_samples=45, algorithm='auto').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# print(len(set(labels)))
print(set(labels))
# set labels gives number like 0 1 2 basically groups
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
'''
df = DataFrame(dict(x=indexes_points[:, 1],
               y=indexes_points[:, 0], label=cluster))
colors = {-1: 'red', 0: 'blue', 1: 'orange',
          2: 'green', 3: 'skyblue'}
fig, ax = plt.subplots(figsize=(8, 8))
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind="scatter", x='x',
               y='y', label=key, color=colors[key])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
'''
# print(indexes_points)

#cv2.imshow("Image", img)
# cv2.waitKey(0)


unique_labels = set(labels)
#colors = ['y', 'b', 'g', 'r']
colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
# print(colors)
for k, col in zip(unique_labels, colors):
    if k == -1:
        # col = 'k'
        col = (0, 0, 0)
    class_member_mask = (labels == k)
    # this is the core points
    xy_core = X[class_member_mask & core_samples_mask]
    '''plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
             markeredgecolor='k',
             markersize=6)'''
    # this is the non core points
    xy_non_core = X[class_member_mask & ~core_samples_mask]
    '''plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
             markeredgecolor='k',
             markersize=6)'''
    xy = np.concatenate((xy_core, xy_non_core))
    # gives a, b,c in a**2+b*x+c form  and it is one dimensional
    polynomial = np.polyfit(xy[:, 0], xy[:, 1], 2)
    poly_value(polynomial[0], polynomial[1], polynomial[2], xy)
    print(polynomial)
    # print(polynomial.ndim)
cv2.imshow("Hi", img)
cv2.waitKey(0)
plt.title('number of clusters: %d' % n_clusters_)
#sc = metrics.silhouette_score(X, labels)
#print("Silhouette Coefficient:%0.2f" % sc)
#ari = adjusted_rand_score(y_true, labels)
#print("Adjusted Rand Index: %0.2f" % ari)
# plt.show()
