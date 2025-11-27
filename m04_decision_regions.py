from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # 마커와 컬러맵 설정
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # 고유한 y값 개수만큼 색깔 지정

    # 전체 그래프 크기 설정
    # 최솟값보다 1 작게, 최댓값보다 1 크게
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 첫 번째 특성
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 두 번째 특성

    # meshgrid(): 2차원 평면에 최댓값과 최솟값(범위) 지정해주면 
    # 이 안에 들어가는 모든 좌표의 x값과 y값을 반환해준다
    # 각 지점에 마킹하거나 색을 칠할 때 자주 쓴다
    # resolution: 간격 (0.02이므로 색깔이 칠해져 있는 것처럼 보인다)
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # rabel(): 일차원으로 펼쳐주는 함수 (predict 함수 사용하기 위해서)
    # classifier는 모델(퍼셉트론 등)의 객체
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)  # xx1 사이즈(2차원)로 재변환
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)  # lab에 따라 색깔을 칠한다
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx], 
                    label=f'Class {cl}', edgecolors='black')