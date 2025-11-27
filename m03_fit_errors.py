import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from m01_perceptron import Perceptron
from m04_decision_regions import plot_decision_regions

# 붓꽃 데이터셋 로드
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')

# setosa, versicolor를 선택하여 y에 넣기
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)  # setosa이면 0, versicolor면 1로 넣기
# 꽃받침 길이(0), 꽃잎 길이(2) 추출해서 X에 넣기
X = df.iloc[0:100, [0, 2]].values

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

# 에포크 대비 잘못 분류된 오차의 그래프 그리기
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()


# 붓꽃 데이터셋 결정 경계 그리기
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()