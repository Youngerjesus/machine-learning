# Train the model with gradient descent

- linear regression 에서 최적의 w 와 b 를 찾을 수 있는 방법으로 gradient descent 가 있다. 
  - cost function 에서 최솟값을 가지도록 w 와 b 를 설계하는 방법이겠지. 
- 이 gradient descent 는 deep learning 의 neural network 에서도 사용된다.
  - general 하게 사용된다고 함. 
  - two parameter 이상을 가진 model 에서.
    - ex) J(w1, w2, w3, ... wn, b) 의 cost function 에서 parameter 는 n + 1 인데 이런 곳에서도 사용할 수 있다. 
- 시작은 초기값 예측부터 한다. w 와 b 둘 다 0 으로 주면서. 
  - 그 다음으로는 최솟값 근처로 갈 때까지 게속해서 w 와 b 값을 바꿔가보는 것이다.
  - 중요한 건 **활 모양인 함수가 아닌 경우에는 최소 지점이 하나 이상일 수 있다는 걸 아는 것이다.** 
    - 물론 linear regression 에서는 활 모양의 함수만 보게 될 것이지만..

![](../images/other%20function%20.png)

- 활 모양이 아닌 함수. 뉴럴 네트워크에서는 이런 함수들을 볼 수도 있다.
- 마치 골프장의 모습이다 hill 과 valley 가 있는. 
- gradient descent 는 언덕위에 올라가서 관찰해보면서 valley 를 찾아 내려가는 걸 말한다. (cost function 이 최소화 되는 지점을 찾으려고.)
  - hill 에서 주위를 둘러보면서 어떤 방향이 가장 가파른지를 찾는거지. 
  - 이렇게 도달하다보면 local minimum 에 도달한다. local 이 붙은 이유가 이거 말고 다른 최저점도 있어서.

## Implementing gradient descent 

![](../images/gradient%20dexcent%20algorithm.png)

- 왼쪽의 w 와 b 가 경사를 내려가는 다음 스텝이다. 이게 수렴할 때까지, local minimum 에 도달할 때까지 이동한다.
  - 중요한 건 w 와 b 는 동시에 업데이트 해야된다는 것. (파라미터들을. 왜그렇지?)
- 여기서 a 는 learning rate 라고 한다. 작은 값임. 0과 1 사이의. (작은 스텝을 위해서 존재한다.)

![](../images/gradient%20descent%20algorithm%20correct,%20incorrect.png)

## Gradient descent intuition

- 이전에 설명이 부족했던 미분을 왜하고 learning rate 를 왜 곱해서 parameter 를 업데이트 하는지 설명.
  - derivative 가 아니라 사실 partial derivative 임. 

![](../images/gradient%20descent%20meaning.png)

- gradient descent 식이 의미하는 바를 보여주는 그림(?), 식이다. 
- 원래 w 와 b 로 이뤄진 다차원 변수의 미분에서 변수 한 개로 나눠서 설명한 것.
- learning rate * w 의 미분은 해당 접선 방향으로 내려가는 걸 의미한다. 이렇게 내려가다 보면 최소지점을 만나는거고. 식이 의미하는 바가 이거임.
  - 최소 지점을 향해서 접선 방향으로 내려간다.

## Learning rate 

- learning rate 는 왜 그 값으로 선택되었을까? 
- 이걸 잘 선택하는게 중요하다고 한다. 
  - learning rate 가 too small 이라면?
    - gradient descent 는 잘 작동할 것이다. 그러나 너무너무너무너무 느릴 것.
  - learning rate 가 too big 이라면?
    - 너무 big 스텝을 가서 현재 시점보다 cost function 이 더 커질 수도 있다. (그러면 big 으로 가다가 small 로 전환하면 되지 않을까?)
    - 그래서 minimum 에 도달하지 못할 수 있음.
- 이미 minimum 지점이라면 parameter 는 업데이트 되지 않는다. 미분이 0 이라서.
  - 이것이 gradient descent 가 local minimum 에 도달할 수 있는 이유로 설명이 된다. (도달한다면 멈출 것이니까.)
    - 그리고 또 다른 이유로는 local minimum 에 도달하기 직전에 미분 값은 굉장히 작아지기 시작할 것이다. 즉 천천히 다가가다 보면 도달할 수 있다. 
      - 이건 learning rate 의 값을 줄이지 않아도, 고정값이어도 가능하다.

## Gradient descent for linear regression

- linear regression 에서 어떻게 gradient descent 를 적용하는지 

![](../images/graident%20descent%20for%20linear%20regression.png)

![](../images/gradient%20descent%20for%20linear%20regression%20explain.png)

## Running gradient descent

- gradient descent 를 실제로 linear regression 에 적용하는 그림.
- gradient descent 는 batch gradient descent 라고도 불린다. 각각의 스텝이 모든 training set 의 데이터를 이용하니까.
  - 물론 other gradient descent 도 있다. training set 의 subset 을 이용하는.
  - linear regression 은 batch gradient descent 를 씀.

*** 

# Optional Lab: Gradient Descent for Linear Regression

## Goals

In this lab, you will:
- automate the process of optimizing $w$ and $b$ using gradient descent.

## Tools

In this lab, we will make use of:
- NumPy, a popular library for scientific computing
- Matplotlib, a popular library for plotting data
- plotting routines in the lab_utils.py file in the local directory

```jupyterpython
import math, copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
```

# Problem Statement

Let's use the same two data points as before - a house with 1000 square feet sold for \\$300,000 and a house with 2000 square feet sold for \\$500,000.

| Size (1000 sqft)     | Price (1000s of dollars) |
| ----------------| ------------------------ |
| 1               | 300                      |
| 2               | 500                      |

```jupyterpython
# Load our data set
x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value
```

### Compute_Cost
This was developed in the last lab. We'll need it again here.

```jupyterpython
#Function to calculate the cost
def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost
```

## Gradient descent summary
So far in this course, you have developed a linear model that predicts $f_{w,b}(x^{(i)})$:
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$
In linear regression, you utilize input training data to fit the parameters $w$,$b$ by minimizing a measure of the error between our predictions $f_{w,b}(x^{(i)})$ and the actual data $y^{(i)}$. The measure is called the $cost$, $J(w,b)$. In training you measure the cost over all of our training samples $x^{(i)},y^{(i)}$
$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2\tag{2}$$ 

In lecture, *gradient descent* was described as:

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{3}  \; \newline
b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}$$
where, parameters $w$, $b$ are updated simultaneously.  
The gradient is defined as:
$$
\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{4}\\
\frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{5}\\
\end{align}
$$

Here *simultaniously* means that you calculate the partial derivatives for all the parameters before updating any of the parameters.

## Implement Gradient Descent
You will implement gradient descent algorithm for one feature. You will need three functions.
- `compute_gradient` implementing equation (4) and (5) above
- `compute_cost` implementing equation (2) above (code from previous lab)
- `gradient_descent`, utilizing compute_gradient and compute_cost

Conventions:
- The naming of python variables containing partial derivatives follows this pattern,$\frac{\partial J(w,b)}{\partial b}$  will be `dj_db`.
- w.r.t is With Respect To, as in partial derivative of $J(wb)$ With Respect To $b$.

### compute_gradient
<a name='ex-01'></a>
`compute_gradient`  implements (4) and (5) above and returns $\frac{\partial J(w,b)}{\partial w}$,$\frac{\partial J(w,b)}{\partial b}$. The embedded comments describe the operations.

```jupyterpython
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db
```

The lectures described how gradient descent utilizes the partial derivative of the cost with respect to a parameter at a point to update that parameter.   
Let's use our `compute_gradient` function to find and plot some partial derivatives of our cost function relative to one of the parameters, $w_0$.

```jupyterpython
plt_gradients(x_train,y_train, compute_cost, compute_gradient)
plt.show()
```

Above, the left plot shows $\frac{\partial J(w,b)}{\partial w}$ or the slope of the cost curve relative to $w$ at three points. On the right side of the plot, the derivative is positive, while on the left it is negative. Due to the 'bowl shape', the derivatives will always lead gradient descent toward the bottom where the gradient is zero.

The left plot has fixed $b=100$. Gradient descent will utilize both $\frac{\partial J(w,b)}{\partial w}$ and $\frac{\partial J(w,b)}{\partial b}$ to update parameters. The 'quiver plot' on the right provides a means of viewing the gradient of both parameters. The arrow sizes reflect the magnitude of the gradient at that point. The direction and slope of the arrow reflects the ratio of $\frac{\partial J(w,b)}{\partial w}$ and $\frac{\partial J(w,b)}{\partial b}$ at that point.
Note that the gradient points *away* from the minimum. Review equation (3) above. The scaled gradient is *subtracted* from the current value of $w$ or $b$. This moves the parameter in a direction that will reduce cost.

###  Gradient Descent
Now that gradients can be computed,  gradient descent, described in equation (3) above can be implemented below in `gradient_descent`. The details of the implementation are described in the comments. Below, you will utilize this function to find optimal values of $w$ and $b$ on the training data.

```jupyterpython
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing
```

```jupyterpython
# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
```

Take a moment and note some characteristics of the gradient descent process printed above.

- The cost starts large and rapidly declines as described in the slide from the lecture.
- The partial derivatives, `dj_dw`, and `dj_db` also get smaller, rapidly at first and then more slowly. As shown in the diagram from the lecture, as the process nears the 'bottom of the bowl' progress is slower due to the smaller value of the derivative at that point.
- progress slows though the learning rate, alpha, remains fixed

### Cost versus iterations of gradient descent 

A plot of cost versus iterations is a useful measure of progress in gradient descent. Cost should always decrease in successful runs. The change in cost is so rapid initially, it is useful to plot the initial decent on a different scale than the final descent. In the plots below, note the scale of cost on the axes and the iteration step.

```jupyterpython
# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()
```

### Predictions
Now that you have discovered the optimal values for the parameters $w$ and $b$, you can now use the model to predict housing values based on our learned parameters. As expected, the predicted values are nearly the same as the training values for the same housing. Further, the value not in the prediction is in line with the expected value.

````jupyterpython
print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")
````

## Plotting
You can show the progress of gradient descent during its execution by plotting the cost over iterations on a contour plot of the cost(w,b). 

````jupyterpython
fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)
````

Above, the contour plot shows the $cost(w,b)$ over a range of $w$ and $b$. Cost levels are represented by the rings. Overlayed, using red arrows, is the path of gradient descent. Here are some things to note:
- The path makes steady (monotonic) progress toward its goal.
- initial steps are much larger than the steps near the goal.

**Zooming in**, we can see that final steps of gradient descent. Note the distance between steps shrinks as the gradient approaches zero.

````jupyterpython
fig, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5],
            contours=[1,5,10,20],resolution=0.5)
````

### Increased Learning Rate

In the lecture, there was a discussion related to the proper value of the learning rate, $\alpha$ in equation(3). The larger $\alpha$ is, the faster gradient descent will converge to a solution. But, if it is too large, gradient descent will diverge. Above you have an example of a solution which converges nicely.

Let's try increasing the value of  $\alpha$ and see what happens:

````jupyterpython
# initialize parameters
w_init = 0
b_init = 0
# set alpha to a large value
iterations = 10
tmp_alpha = 8.0e-1
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
````

Above, $w$ and $b$ are bouncing back and forth between positive and negative with the absolute value increasing with each iteration. Further, each iteration $\frac{\partial J(w,b)}{\partial w}$ changes sign and cost is increasing rather than decreasing. This is a clear sign that the *learning rate is too large* and the solution is diverging. 

````jupyterpython
plt_divergence(p_hist, J_hist,x_train, y_train)
plt.show()
````

Above, the left graph shows $w$'s progression over the first few steps of gradient descent. $w$ oscillates from positive to negative and cost grows rapidly. Gradient Descent is operating on both $w$ and $b$ simultaneously, so one needs the 3-D plot on the right for the complete picture.


