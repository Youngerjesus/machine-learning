# Gradient descent in practice 

## Feature scaling 

- feature scaling 을 통해서 gradient descent 가 더 빨라지게 할 수 있다.  
- feature 의 range 범위가 차이가 많이나면 gradient descent 가 느리다. 그래서 comparable 한 수준으로 transformation 하는 Rescaling 을 하면 도움이 된다.
  - feature 의 범위가 큰 게 있고 작은게 있다면 일반적으로 범위가 큰 쪽의 파라미터는 작을 거고, 범위가 작은 쪽의 파라미터는 클 것. 이 경우에서 gradient descent 를 실행하면 불필요하게 하강되는 범위가 크거나 작을 수 있다. 
  - 그래서 rescale 해서 feature 의 범위를 0~1 사이로 맞춰두는 것. 그러면 더 빠를 것.

![](../images/feature%20rescale.png)


- rescle 하는 과정은 최대값으로 나누는 것이다. x1 의 범위가 $300 <= x1 <= 2000$ 이라면 2000으로 나누고, x2 의 범위가 $0 <= x2 <= 5$ 라면 5로 나눠서 조정하는 것. 
- 다른 방식으로는 mean normalization 을 적용할 수 있다. 

![](../images/mean%20normalization.png)

- 마지막 방식으로는 Z-score normalization 도 있다. 
  - 정규분포와 표준편차를 이용해서 scaling 을 조정하는 것. 

![](../images/z-score%20normalization.png)

- feature scaling 할 때 약간 여유루워도 된다고 함. 
  - x1 이 $-1 <= x1 <= 1$ 인 경우 다른 feature 이 $-0.1 <= xi <= 1$ or $-3 <= xi <= 3$ 이어도 괜찮댜고 한다.
  - 근데 $ -100 <= xi <= 100$ 정도면 rescalie 이 필요하다고 한다. 

## Checking gradient descent for convergence 

- gradient descent 가 수렴하고 있는지 어떻게 아는지를 말하는 것. 즉 global minimum 에 가까운 parameter 를 정하는 것.  

- 머신러닝에 사용되는 learning curve 는 이런 것만 있다. 

![](../images/learning%20curve.png)

- gradient descent iteration 이후에 cost function 값이 올랐다면 그건 learning rate Alpha 를 잘못 선택한 것이나 코드에 버그가 있는 것.
- learning curve 가 flatten 해졌다면 converge 된 것. (더이상 작아지지 않는다면.)
  - 아니면 automatic convergence test 를 통해서 converge 되었는지 확인할 수 있다. 
    - iteration 이후에 0.001 보다 작게 cost function 이 감소되었다면 converge 되었다고 하는 것. 


## Choosing the learning rate

- gradient descent 를 Iteration 돌리면 cost function 이 점점 작아져야한다. 만약 이 경우가 아니라면 코드에 버그가 있거나, learning rate 가 너무 큰 것. 
- 그래서 저자는 이를 이용한 약간의 팁으로 gradient descent 가 수렴하지 않느다면 learning rate 를 극도로 작게 줘서 확인한다고 함.
- learning rate 를 고를 때 0.001 부터 시작해서 x3 씩 하면서 올려나간다고 한다. cost function 이 계속해서 감소하는 한도 내에서.

## Feature engineering 

- feature 를 고르는 것은 모델의 성능을 결정하는데 중요한 요소이다.
- 예로 집의 가격을 예측하는데 집의 가로 길이인 frontage 와 세로 길이인 depth 를 feature 로 가졌다고 해보자. 그럼 이 둘의 곱을 가진 area 라는 새로운 feature 를 만들어 볼 수 있다.
  - 새로운 feature 를 만들어냈는데 이게 더 중요한 요소일 수 있다. 

## Polynomial regression 

- 여기선 data 를 보고 straight line 으로 예측하는게 아니라 non-linear function 인 curve 와 같은 function 으로 예측하는 것을 다룸.

![](../images/polynomial%20regression%20feature%20scaling.png)

- x^2 과 x^3 같이 있는 다항함수에선 Feature scaling 이 훨씬 중요해진다. gradient descent 에서 불필요한 엄청나게 많은 변화를 줄 수 있기 떄문에.
- 여기서는 각 feature 과 target 데이터를 그래프로 보면서 feature 이 어떤 성향을 가지고 있는지 확인하면서 정하는 일을 하네. linear 한지, non-linear 한 지.
  - 이게 polynomial function 을 사용하는 것. 
- 여기서 feature engineering 도 같이 적용.
