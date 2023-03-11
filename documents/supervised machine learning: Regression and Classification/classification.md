# Classification 

## Motivation 

- linear regression 은 classification 에 적용하기에는 좋은 알고리즘이 아니다. 
  - 실제로 적용해보면 데이터셋이 추가될 때 모델의 정확성이 떨어진다.  

- Classification 의 예시를 먼저 보자. 
  - Is this email spam? 
  - Is the transaction fraudulent? 
  - is the tumor malignant? 
  - 이렇게 예측하는 output 이 yes or no 인 것들을 binary classification 이라고 한다.

- 이런 binary classification 을 푸는 것으로 logistic regression 이 있다. 

## Logistic Regression 

- logistic regression 은 0 과 1 을 나타내는 분류하는 경우에 적합하다. 그리고 실제로 구현하는데 중요한 수학적 함수들이 있다. sigmoid function (= logistic function )

![](../images/logistic%20regression.png)

- sigmoid function 에서 z 값이 커지면 1 에 가까워지고 z 값이 엄청 작다면 0 에 가까워진다. 
- sigmoid function 의 y 값은 확률을 나타낸다고 보면된다.  

- 다음은 logistic regression 알고리즘을 적용하는 방법

![](../images/Implement%20logistic%20regression.png)

- Linear regression 의 식을 이용한다. 이 식을 signoid function 에다가 대입하는 것. 이개 logistic regression 이다. 
  - feature 와 파라미터를 가진 것을 어느 함수에다 적용할 지 결정하는게 ML 인가 싶다.  

- logistic regression 이 어떻게 쓰이는지는 확률을 생각해보면 된다. 1 이 나올 확률, 0 이 나올 확률. 

![](../images/interpretation%20of%20logistic%20regression%20output.png)

- 여기선 70% 확률로 1이다 라고 하는 것. 

## Decision Boundary 

- Decision Boundary 를 통해서 logistic regression 이 어떻게 계산하는지 알 수 있다. 

![](../images/logistic%20regresssion%20predict.png)

- logistic regression 에서는 확률로 내놓는다고 했는데 결국엔 0 과 1 을 예측해야한다. 그래서 threshold 값이 0.5 로 정해놓고 이 값보다 크다면 1 로 예측하고 아니라면 0 으로 예측하면 된다.
- 이걸 더 정리하다 보면 $w * x + b > 0$ 이면 1 이고 0보다 작다면 0으로 예측한다. 
  - w 와 x 는 여기서 vector 값.
  - 그리고 $z = w * x + b = 0$ 인 곳을 decision boundary 라고 한다.  

![](../images/decision%20boundary.png)

- logistic regression 은 decision boundary 를 찾는거네. linear regression 처럼 파라미터들을 찾는 것. 

![](../images/decision%20boundary%202.png)

- 이 함수보다 더 복잡한 decision boundary 도 있다. (polynomial feature 를 여기서도 쓸 수 있네.)
- logistic regression 은 결국에 decision boundary (0.5 를 나타내는 함수) 를 찾는 문제로 바뀐다.
