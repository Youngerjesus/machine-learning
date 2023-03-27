# Vectorization 

## How neural networks are implemented efficiently

- Neural Network 는 vector 화를 통해서 행렬 곱셈 연산을 효율적으로 할 수 있다. 이게 큰 신경망을 구축하는데 많은 영향을 줌. 
  - 병렬 컴퓨팅 하드웨어나 GPU, CPU 같은 것들이 이런 연산에 도움을 준다. 
  - 이런 계산력이 되게 중요하다고 하네. 

![](../images/for%20loop%20ve.%20vecotrization.png)

- For loop 으로 하나하나 계산하는게 아니라 np.matmul(a_in, w) 라는 함수 호출을 통해서 한번에 행렬 계산을 할 수 있다.
  - matmul 이 matrix multiplication 이라는 뜻임.
- vectorization 은 위의 matmul 게산을 말하고 이게 어떻게 동작하는지는 다음 강의에서 다룸. 

## Matrix multiplication

- transpose 라는 과정이 있는 것도 알아두자. column 기반에서 row 기반으로 바뀌는 것.
- 계산 과정을 자세하게 보자. 

![](../images/vector%20matrix%20multiplication.png)

![](../images/matrix%20matrix%20multiplication.png)

## Matrix multiplication rules

- 행렬 곱셈을 할 때 전치 행렬 (transpose) 를 이용해서 쓴다는 것.
  - activation 데이터나 input 데이터는 column 기반으로 되어있어서 계산하려면 transpose 해줘야함.
- 행렬 곱셈은 두 행렬의 열과 행이 같을 때만 적용할 수 있다는 것.

![](../images/matrix%20multiplication%20rule.png) 


## Matrix multiplication code

![](../images/matmul%20result.png)

![](../images/dense%20layer%20vectorized.png)
