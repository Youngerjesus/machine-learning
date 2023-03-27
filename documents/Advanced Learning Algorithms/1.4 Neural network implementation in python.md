## Forward prop in a single layer

## Forward prop in a single layer

![](../images/forward%20prop%20in%20a%20single%20layer.png)

- forward propagation 을 하드코딩해서 어떻게 동작하는지 보자.  
  - Tensorflow 나 pytorch 와 같은 프레임워크에서 어떻게 동작하는지.
- 각 뉴런에 대해서 값을 계산해서 a1_1, a1_2, a1_3 을 계산해서 a vector 를 만들어서 이를 다음 계층에 전달.
  - a1 = [a1, a2, a3] 가 되는 거임.
- a1_1 이나 a1_2 가 어떤 표현 용어인지 알아야 하고, w1_1, b1_1 을 이용해서 z1_1 을 계산하고 이를 이용해서 a1_1 을 게산한다. 그리고 a1_1 과 a1_2, a1_3 이 모여서 a vector 가 되고.
- vector 들의 곱을 쓰는게 dot 연산이었음.

## General implementation of forward propagation

![](../images/dense%20and%20sequential.png)

- 더 일반적인 forward propagation 구현에 대해서 알아보는 것.
  - dense 함수와 sequential 함수가 내부적으로 어떻게 되어있는지 보자.  
  - dense 함수를 통해서 신경망의 하나의 layer 를 구축하는 것. 이 함수는 이전 layer 의 출력 값과 w 와 b 를 입력 받아서 현재 계층의 활성화 값을 출력한다.
    - (w 와 b 값은 내부적으로 만들어지는게 아니라 입력 받는 건가?)
      - backpropagation 과 gradient descent 에 의해서 결정된다고 함.
      - 과정은 이럼.
        - 1) w 와 b 를 무작위로 초기화
        - 2) 데이터를 네트워크에 전달해서 forward propagation 을 통해서 계산함.
        - 3) 손실 함수 (Loss function) 을 통해서 출력과 오차를 계산.
        - 4) backpropagation 을 통해서 w (가중치) 와 b (편향) 에 대한 기울기 (gradient) 를 계산한다. 오차가 감소하는 방향을 찾기 위해서가 목적. 
        - 5) gradient descent 를 통해서 w 와 b 를 업데이트한다. loss function 이 최소화 되도록 하기 위해서. 
        - 6) 여러 epoch 동안 위의 과정을 반복한다. 각 에포크마다 전체 데이터 셋을 이용함.
          - epoch 는 전체 train data 가 모델에 한번 완전히 통과되는 것을 말함. 모든 훈련 데이터가 모델에 통과되서 파라미터들이 조절되는 걸 말함.
          - epoch 가 너무 많으면 과적합 될 우려도 있음.
    - 또 dense 에서 g function 도 구현해서 입력받도록 할 수도 있다.
  - dense 를 통해서 forward propagation 이 구현이 된다. 

- 용어 정리에 대해서도 배우자면 W 와 같은 대문자는 matrix 를 의미하고, w 와 같은 소문자는 vector 나 scalar 값을 의미한다. 

- 이걸 배우는 이유가 내부 동작을 알면 코드 디버깅에 도움이 되고 깊은 이해를 주고 싶어서.


