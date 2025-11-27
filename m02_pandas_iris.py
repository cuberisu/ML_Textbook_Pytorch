import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 붓꽃 데이터셋을 DataFrame 객체로 로드하기
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# csv 파일 읽기
df = pd.read_csv(s, header=None, encoding='utf-8')
# print(df.tail())

# 붓꽃 데이터셋은 인덱스 4번째 열(클래스 레이블)이 setosa, versicolor, virginica인 샘플이
# 각각 50개씩 순서대로 존재

# setosa, versicolor를 선택하여 y에 넣기
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)  # setosa이면 0, versicolor면 1로 넣기

# 꽃받침 길이(0), 꽃잎 길이(2) 추출
X = df.iloc[0:100, [0, 2]].values

# 산점도 그리기
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')  # x축 꽃받침 길이 cm
plt.ylabel('Petal length [cm]')  # y축 꽃잎 길이 cm
plt.legend(loc='upper left')
plt.show()