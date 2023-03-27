## Neural network layer

- 대부분의 Neural Network 는 뉴런들의 layer 로 이뤄진다.
- 오늘 날의 신경망 네트워크는 몇백개 된다는 듯.
- 여기서는 동작 방법에 대해서 알려주려고 했던 듯.
  - input vector 가 각 뉴런에게 전달되고, 각 뉴런에 있는 파라미터에 의해서 logistic regression 계산 된 다음에 다음 layer 에게 전달될 때 vector 형식으로 전달됨.
![](../images/layer%20denote.png)

- 레이어가 여러개있을 때 파라미터와 레이어 표기법.
- 뉴런을 hidden layer 의 unit 이라고 부르기도 하네.

![](../images/layer%20denote2.png)

- output layer 에서 파라미터와 출력값 표기법. 
- 여기서 출력값은 scala 값이다.

## More complex neural networks

![](../images/layer%20denote%20summarization.png)

- 정리하면 이렇게 될 듯. 
- 여기서 쓰는 signoid function 은 activation function 으로도 얘기함.
- x vector 는 a[0] 로 대신 쓸 수 있음. 

## Inference: making predictions (forward propagation)

![](../images/forward%20propagation.png)

- forward propagation 이 뭔지. 
  - activation 을 전파하는 것. 계산할 때 사용한다.
  - backward propagation 이라는 것도 있다.
    - 이건 learning 에 사용됨. 

- 그리고 위의 Neural network 와 같이 초기에 unit or neuron 이 많고 점점 레이어가 갈수록 unit 이 줄어드는게 전형적인 Neural architecture 이다.
