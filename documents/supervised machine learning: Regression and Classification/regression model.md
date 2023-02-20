# Regression Model 


## Linear Regression 

- 데이터를 가지고 straight line 으로 만드는 것.
- ex) House size 를 가지고 House price 를 예측하는 것.
- 실제로 이걸 쓸 땐 데이터를 가지고 그래프를 그려보고, 직선의 유형을 가지고 있는지 확인해보고 적용하면 될 것 같다.

- training set 을 가지고 learning algorithm  에 입력을 하면 function 을 만든다. 
  - 이런 function 을 hypothesis (가설, 가정) 이라고 한다. model 이라고도 부른다.
  - 여기 이 function 에다가 새로운 input 을 넣으면 예측을 해준다. 이 때의 예측 y 값을 `y-hat` 이라고한다. (training set 의 y 값을 target 이라고 했었다. 이 예측값이랑은 조금 이름이 다름.)
  - 이 learning algorithm 이 Linear regression 이 아닌가.

- 자 이제 function 을 어떻게 만들 것인가? 수학식을 써서.
  - function 은 straight line 을 그린다고 한다. 이거에 맞춰서 식이 나와야함. 
  - $f_[w,b](x) = wx + b$ 여기서 w 와 b 를 정해주면 예측값 y 를 만들 수 있다.
  - 물론 linear function 이 아니라 curve function 을 예측해볼 수 있다. 여기서는 simple 해서 이걸 소개함. (실제로 현실의 모델들은 linear 가 아닌 경우가 많겠지.)

- one input or feature 를 가지고 linear regression 을 하는 걸 univariate (일변량의) linear regression 이라고 한다.
- 물론 이거 말고도 여러개의 Input 을 넣고 linear regression 을 적용해서 function 을 만들 수 있다.

### Terminology 

- Training set: Data used to train the model
- Notation `x`: input or feature or input feature 
- Notation `y`: output or target
- Notation `m`: number of training sets
- Notation `(x, y)`: single training set
- Notation `$(xi, yi)$`: specific `i`th training set

***

## Model Representation lab

Learn to implement the model $f_{w,b}$ for linear regression with one variable

````jupyterpython
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
````

- NumPy, a popular library for scientific computing
- Matplotlib, a popular library for plotting data


### Problem Statement

As in the lecture, you will use the motivating example of housing price prediction.  
This lab will use a simple data set with only two data points - a house with 1000 square feet(sqft) sold for \\$300,000 and a house with 2000 square feet sold for \\$500,000. These two points will constitute our *data or training set*. In this lab, the units of size are 1000 sqft and the units of price are 1000s of dollars.

You would like to fit a linear regression model (shown above as the blue straight line) through these two points, so you can then predict price for other houses - say, a house with 1200 sqft.

### Number of training examples `m`

You will use `m` to denote the number of training examples. Numpy arrays have a `.shape` parameter. `x_train.shape` returns a python tuple with an entry for each dimension. `x_train.shape[0]` is the length of the array and number of examples as shown below.

`````jupyterpython
# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
`````

- x_train.shape: (2,)
- Number of training examples is: 2

One can also use the Python `len()` function as shown below.

````jupyterpython
# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")
````

- Number of training examples is: 2

### Training example `x_i, y_i`

You will use (x$^{(i)}$, y$^{(i)}$) to denote the $i^{th}$ training example. Since Python is zero indexed, (x$^{(0)}$, y$^{(0)}$) is (1.0, 300.0) and (x$^{(1)}$, y$^{(1)}$) is (2.0, 500.0).

To access a value in a Numpy array, one indexes the array with the desired offset. For example the syntax to access location zero of `x_train` is `x_train[0]`.
Run the next code block below to get the $i^{th}$ training example.

````jupyterpython
i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
````

- (x^(0), y^(0)) = (1.0, 300.0)

### Plotting the data

You can plot these two points using the `scatter()` function in the `matplotlib` library, as shown in the cell below.
- The function arguments `marker` and `c` show the points as red crosses (the default is blue dots).

You can use other functions in the `matplotlib` library to set the title and labels to display

````jupyterpython
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()
````

### Model function

As described in lecture, the model function for linear regression (which is a function that maps from `x` to `y`) is represented as

$$ f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$

The formula above is how you can represent straight lines - different values of $w$ and $b$ give you different straight lines on the plot. <br/> <br/> <br/> <br/> <br/>

Let's try to get a better intuition for this through the code blocks below. Let's start with $w = 100$ and $b = 100$.

**Note: You can come back to this cell to adjust the model's w and b parameters**

````jupyterpython
w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")
````

- w: 100
- b: 100

Now, let's compute the value of $f_{w,b}(x^{(i)})$ for your two data points. You can explicitly write this out for each data point as -

for $x^{(0)}$, `f_wb = w * x[0] + b`

for $x^{(1)}$, `f_wb = w * x[1] + b`

For a large number of data points, this can get unwieldy and repetitive. So instead, you can calculate the function output in a `for` loop as shown in the `compute_model_output` function below.
> **Note**: The argument description `(ndarray (m,))` describes a Numpy n-dimensional array of shape (m,). `(scalar)` describes an argument without dimensions, just a magnitude.  
> **Note**: `np.zero(n)` will return a one-dimensional numpy array with $n$ entries   

````jupyterpython
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb
````

Now let's call the `compute_model_output` function and plot the output..

````jupyterpython
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
````

### Prediction

Now that we have a model, we can use it to make our original prediction. Let's predict the price of a house with 1200 sqft. Since the units of $x$ are in 1000's of sqft, $x$ is 1.2.

````jupyterpython
w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")
````

- $340 thousand dollars

***

## Cost function formula 

- Linear regression 을 구현하기 위해서 해야하는 것으로 cost function 을 정의해야한다.
  - 트레이닝 셋과 가장 오차가 적은 가설 함수 (hypothesis) 를 도출하기 위한 것.
- cost function 은 모델이 얼마나 잘 작동하는지 알려준다.
- 아까 나왔던 f_[w,b](x) = wx + b$ 에서 w 와 b 를 parameter of model 이라고 한다. 이 파라미터를 training 중에 측정하고 이게 모델의 성능을 좌우할 것이다.
  - 그리고 w 를 weight (가중치) 라고 부르고 b 를 coefficient (계수) 라고 부른다.
- 어떻게해야 w 와 b 를 잘 설계할 수 있을까? 

- **cost function 을 설계함으로써 우리 데이터셋에 맞는 최적의 함수를 내놓을 수 있게 된다.**

![](../images/cost%20function.png)

- error = (y-hat - y)
- 에러에서 제곱을 하는 이유는 그냥 계산하면 0 이 나오니까. 양수로 만들려고.
- m 으로 나눈 이유는 training set 이 많아질수록 cost function 이 커질 것이니까 그거 막을려고 
- 2m 으로 나눈 이유는 계산을 깔끔하게 하려고 미분할 때. 
- 이렇게 만든 cost function 을 `the square error cost function` 이라고한다. 
  - 머신러닝에서 사람들은 각각 다른 cost function 을 사용한다. 각기 다른 최적화를 한다. 
  - linear regression 에선 `the square error cost function` 이걸로 최적화를 하는 거고.


## Cost function intuition 

- cost function 을 제일 작게 만들도록 하면 model 의 파라미터인 (= 이 예시에선 w 와 b) 를 설계할 수 있다.
- cost function 의 파라미터는 w 와 b 일건데 이 그래프의 최솟값이 결국 실제 데이터와 예측치 값의 최소값이니까 제일 정확한게 나오겠지.

![](../images/cost%20function%20visualization.png)

- gradient descent 를 통해서 이런 변수 두 개가 있는 cost function 의 최솟값을 구할 수 있다.

## Cost function 실습 

### Goal

In this lab you will:
- you will implement and explore the `cost` function for linear regression with one variable. 

`````jupyterpython
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')
`````

- NumPy, a popular library for scientific computing
- Matplotlib, a popular library for plotting data
- local plotting routines in the lab_utils_uni.py file in the local directory

### Problem Statement

You would like a model which can predict housing prices given the size of the house.  
Let's use the same two data points as before the previous lab- a house with 1000 square feet sold for \\$300,000 and a house with 2000 square feet sold for \\$500,000.


| Size (1000 sqft)     | Price (1000s of dollars) |
| -------------------| ------------------------ |
| 1                 | 300                      |
| 2                  | 500                      |

````jupyterpython
x_train = np.array([1.0, 2.0])           #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])           #(price in 1000s of dollars)
````

### Computing Cost
The term 'cost' in this assignment might be a little confusing since the data is housing cost. Here, cost is a measure how well our model is predicting the target price of the house. The term 'price' is used for housing data.

The equation for cost with one variable is:
$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$

where
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}$$

- $f_{w,b}(x^{(i)})$ is our prediction for example $i$ using parameters $w,b$.
- $(f_{w,b}(x^{(i)}) -y^{(i)})^2$ is the squared difference between the target value and the prediction.
- These differences are summed over all the $m$ examples and divided by `2m` to produce the cost, $J(w,b)$.
>Note, in lecture summation ranges are typically from 1 to m, while code will be from 0 to m-1.

The code below calculates cost by looping over each example. In each loop:
- `f_wb`, a prediction is calculated
- the difference between the target and the prediction is calculated and squared.
- this is added to the total cost.

````jupyterpython
def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost
````

### Cost Function Intuition 

The cost equation (1) above shows that if $w$ and $b$ can be selected such that the predictions $f_{w,b}(x)$ match the target data $y$, the $(f_{w,b}(x^{(i)}) - y^{(i)})^2 $ term will be zero and the cost minimized. In this simple two point example, you can achieve this!

In the previous lab, you determined that $b=100$ provided an optimal solution so let's set $b$ to 100 and focus on $w$.

<br/>
Below, use the slider control to select the value of $w$ that minimizes cost. It can take a few seconds for the plot to update.


````jupyterpython
plt_intuition(x_train,y_train)
````

The plot contains a few points that are worth mentioning.

- cost is minimized when  𝑤=200 , which matches results from the previous lab
- Because the difference between the target and pediction is squared in the cost equation, the cost increases rapidly when  𝑤 is either too large or too small.
- Using the w and b selected by minimizing cost results in a line which is a perfect fit to the data.

