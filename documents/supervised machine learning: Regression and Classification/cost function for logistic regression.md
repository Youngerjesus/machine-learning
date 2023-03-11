# Cost function for logistic regression 

- Cost function 의 목적은 뭐였는가? 주어진 트레이닝셋을 바탕으로 가장 오차가 적은 파라미터를 선택할 수 있도록 함수를 설계하는 것.
- 여기서는 linear regression 에 쓰였던 the squared error cost function 이 logistic regression 에 적합하지 않다는 걸 보여준다고함.

![](../images/squared%20error%20cost%20for%20logistic%20regression.png)

- squared error cost function 을 적용하면 이렇게 수렴하지 않는 모형이 나온다고 한다. (왜그럼?)
  - - squared error function 에다가 sigmoid function 을 대입해서 그리면 이런 모형이 나와서 (= 수렴이 안되는 모형) 그러지 않을까.
- local minimum 이 굉장히 많은 구조가 됨. 
- 결국에 gradient descent 를 적용하기 위한 구조로 (= convex 한) cost function 이 나오도록 만들면 된다. 이걸 만들도록 loss function 이 등장한다. 
- loss function 이 convex 한 function 이라는 증명은 따로 하지 않음.

![](../images/loss%20function%20on%20single%20training%20set.png)

- 여기서는 loss function 으로 training set 을 적용해보면서 오차가 어떻게 계산되는지 설명해줌. 
  - if y == 1 일 때, y == 0 일 때 우리의 가설 함수가 예측하는 값에 따라서 오차가 어떻게 계산되는지.
  - 예측은 이전에 배운 sigmoid function 을 통해서 한다. 거기서 확률을 기반으로 예측을 했으니까. 
- 올바르게 예측했다면 loss = 0 이다. 근데 잘못 예측할수록 비용은 엄청 커진다. 실제 값은 1 인데 0 에 가깝게 예측했다면 loss 는 엄청남.


![](../images/cost%20function%20for%20logistic%20regression.png)

- cost function 이 Loss function 을 entire training set 에 적용한 것. 
- 용어 정리 
  - loss: target value 와 예측값의 차이
  - cost: 전체 트레이닝 셋의 loss 를 합친 것. 

![](../images/loss%20function.png)

## Simplified Cost Function for Logistic Regression

- loss function 을 정리하면 이렇게 된다. y == 1 일 때, y == 0 일 때 각각 약간 다른 function 을 쓰도록 하면 되서.

![](../images/simplified%20loss%20function.png)


![](../images/simplified%20cost%20function.png)

- 여기서의 cost function 은 statistic 에서 아이디어를 가지고 옴. (maximum likelihood estimation)
