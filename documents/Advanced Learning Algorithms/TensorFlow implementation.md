# TensorFlow Implementation 

## Inference in Code

- TensorFlow 와 PyTorch 가 딥러닝에서 가장 유명한 프레임워크.
- 여기서는 TensorFlow 를 이용해서 어떻게 코드를 짜서 값을 추론하는지에 대한 설명 

![](../images/tensorflow%20inference.png) 

- 물론 이 그림에서 나오지 않은 것도 있음. w 와 b 같은 parameter 를 어떻게 load 하는지.
- inference 는 forward propagation 과정의 코드.

## Data in TensorFlow

- Numpy 와 TensorFlow 의 데이터 표현 방법이 다르다.  
  - Numpy 가 먼저 시작함.
  - Neural Network 를 하려면 데이터가 있어야하는데 표현 방법이 다르니.
- 이전 강좌에서는 1차원 벡터만 이용했지만, TensorFlow 에서는 2차원 행렬로 데이터를 표현한다. 1차원 벡터는 행과 열을 가지지 않지만 2차원 행렬은 가진다.

![](../images/tensorflow%20representation.png)

- tensor 는 데이터 타입이다. matrix 정보를 담고 계산하기 위해서 만든.
- tensor 는 numpy 형식의 array 로 변환 가능.

![](../images/tensorflow%20representation2.png)

- 결과는 1x1 행렬로 표현.

- 대게 numpy 로 데이터를 조작하다가 tensorflow 로 넘길 땐 변환을 해야한다고 함.

## Building a neural network

- tensorflow 로 neural network 를 만드는 방법 

- 이런식으로 layer 를 합칠 수 있다. 

![](../images/sequential%20layer.png)

- 모델을 트레이닝 셋으로 Neural Network 에서 학습시키려면 이렇게 하면 된다. 

![](../images/train%20model.png)

- 디테일한 내용은 다음주. 

- 예측은 이렇게 한다. 

![](../images/model%20predict.png)

- 정리하면 이렇다. 

![](../images/tensorflow%20model.png)
