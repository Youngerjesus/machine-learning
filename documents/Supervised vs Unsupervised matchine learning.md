# Supervised vs Unsupervised machine learning 


## What is machine learning?

- Field of study that gives a computer the ability to learn without being explicitly programming 

- 머신러닝은 학습시킨 데이터 양에 따라서 퍼포먼스가 다르다.

- 크게 머신러닝 알고리즘은 두 가지로 나뉜다. 
  - Supervised Learning 
    - 실제로 real world 에서 많이 쓰는 어플리케이션이다. 실제로 성과도 제일 많이 나오고 있고.
  - Unsupervised learning 
    - Recommender system 
    - Reinforcement learning 

## Supervised Learning 

- 공통적인 알고리즘이 있다. 
  - `X` 라는 input 을 바탕으로 `Y` 라는 output 으로 매핑하는 걸로 학습한다.
- 학습을 시킬 때 답을 주는 알고리즘을 줘야한다.
- 이렇게 학습을 하면 `이전에 보지 못한 입력값 X` 가 주어졌을 때 Y 를 예측해볼 수 있다.

- UseCase 
  - X is email, Y is spam or not spam email 
  - X is audio clip, Y is text script
  - X is English Y is japaneses 
  - X is ad information, you information Y is if you click on that ad or not
  - X is 집 사이즈 Y is 집 값 
    - 이런 예에 적용하는 알고리즘은 Regression 이라고한다. 
    - 무한히 가능한 범위 내에서 정확한 숫자를 정하는 것. 직선을 그릴 수도 있고, 곡선을 그릴 수도 있다. 
    - 이게 대표적인 Supervised learning 이다. 다른 것은 Classification 

### Classification

- 대표적인 예) breast cancer detection. (유방암 검출.)
  - 그냥 종양과 암을 구별. 
  - 종양은 0 으로 암은 1로 데이터셋을 만든다.
  - regression 과 다른 점은 output 이 작은 small number or category or class or output (다 똑같은 용어.) 라는 것.
  - 데이터 input 은 하나가 아니라 여러개가 될 수 있다. 종양인지 암인지 에측하는데 종양의 사이즈 뿐 아니라 나이같은 것도 중요한 데이터라서 이걸 같이 쓸 수도 있음.

## Unsupervised learning 

- output y 가 정답으로 라벨링 되어있지 않다. 그래서 위의 classification 문제로 보면 어떤게 종양인지, 암인지 구별이 안되어있다.
  - 그래서 이런 질문 자체가 들어올 수 없다. 이를 예측할 수 없기 떄문에. 
  - 대신 데이터 셋을 보고 structure 나 Pattern 을 찾도록 한다. 그래서 데이터를 다른 group 이나 cluster 로 구별하는 일을 할 수 있다. 
    - 이걸 clustering algorithm 이라고 한다. 
    - 예로 구글 뉴스. 구글 뉴스는 비슷한 뉴스끼리 묶어서 보여줌.
      - 제목이나 본문을 보고 아 똑같은 단어가 여러개 있네. 이걸로 묶어서 보여줄 수 있겠지. 

- 또 다른 알고리즘으로는 `Anomaly Detection` 이 있다.
  - 일반적이지 않은 이벤트나 트랜잭션 (= 재무 시스템인 경우) 가 발생했을 때 탐지하는 것.

- 이것 말고도 `Dimensionality reduction` 이라는 것도 있다.
  - compress data using fewer numbers
