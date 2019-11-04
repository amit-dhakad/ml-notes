                                 Google Machine Learning Crash Course





## classification model



A type of machine learning model for distinguishing among two or more discrete classes. For example, a natural language processing classification model could determine whether an input sentence was in French, Spanish, or Italian.





## label



In supervised learning, the "answer" or "result" portion of an [**example**](https://developers.google.com/machine-learning/glossary/#example). Each example in a labeled dataset consists of one or more features and a label. For instance, in a housing dataset, the features might include the number of bedrooms, the number of bathrooms, and the age of the house, while the label might be the house's price. In a spam detection dataset, the features might include the subject line, the sender, and the email message itself, while the label would probably be either "spam" or "not spam."





## regression model



A type of model that outputs continuous (typically, floating-point) values.





# Descending into ML



**Linear regression** is a method for finding the straight line or hyperplane that best fits a set of points. This module explores linear regression intuitively before laying the groundwork for a machine learning approach to linear regression.





# Descending into ML: Linear Regression

It has long been known that crickets (an insect species) chirp more frequently on hotter days than on cooler days. For decades, professional and amateur scientists have cataloged data on chirps-per-minute and temperature. As a birthday gift, your Aunt Ruth gives you her cricket database and asks you to learn a model to predict this relationship. Using this data, you want to explore this relationship.

First, examine your data by plotting it:

![Raw data of chirps/minute (x-axis) vs. temperature (y-axis).](https://developers.google.com/machine-learning/crash-course/images/CricketPoints.svg)

**Figure 1. Chirps per Minute vs. Temperature in Celsius.**

As expected, the plot shows the temperature rising with the number of chirps. Is this relationship between chirps and temperature linear? Yes, you could draw a single straight line like the following to approximate this relationship:

![Best line establishing relationship of chirps/minute (x-axis) vs. temperature (y-axis).](https://developers.google.com/machine-learning/crash-course/images/CricketLine.svg)

**Figure 2. A linear relationship.**

True, the line doesn't pass through every dot, but the line does clearly show the relationship between chirps and temperature. Using the equation for a line, you could write down this relationship as follows:



​                                                    $y=mx+b$

where:

- y is the temperature in Celsius—the value we're trying to predict.
- m is the slope of the line.
- x is the number of chirps per minute—the value of our input feature.
- b is the y-intercept.

By convention in machine learning, you'll write the equation for a model slightly differently:



​                                                       $y′=b+w1x1$       

where:

- y′ is the predicted [label](https://developers.google.com/machine-learning/crash-course/framing/ml-terminology#labels) (a desired output).
- b is the bias (the y-intercept), sometimes referred to as w0.
- w1 is the weight of feature 1. Weight is the same concept as the "slope" m in the traditional equation of a line.
- x1 is a [feature](https://developers.google.com/machine-learning/crash-course/framing/ml-terminology#features) (a known input).

To **infer** (predict) the temperature y′ for a new chirps-per-minute value x1, just substitute the x1 value into this model.

Although this model uses only one feature, a more sophisticated model might rely on multiple features, each having a separate weight (w1, w2, etc.). For example, a model that relies on three features might look as follows:



​                                                         $y′=b+w1x1+w2x2+w3x3$





# Descending into ML: Training and Loss



**Training** a model simply means learning (determining) good values for all the weights and the bias from labeled examples. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called **empirical risk minimization**.

Loss is the penalty for a bad prediction. That is, **loss** is a number indicating how bad the model's prediction was on a single example. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater. The goal of training a model is to find a set of weights and biases that have *low* loss, on average, across all examples. For example, Figure 3 shows a high loss model on the left and a low loss model on the right. Note the following about the figure:

- The arrows represent loss.
- The blue lines represent predictions.

![Two Cartesian plots, each showing a line and some data points. In the first plot, the line is a terrible fit for the data, so the loss is high. In the second plot, the line is a a better fit for the data, so the loss is low.](https://developers.google.com/machine-learning/crash-course/images/LossSideBySide.png)

**Figure 3. High loss in the left model; low loss in the right model.**

 



Notice that the arrows in the left plot are much longer than their counterparts in the right plot. Clearly, the line in the right plot is a much better predictive model than the line in the left plot.

You might be wondering whether you could create a mathematical function—a loss function—that would aggregate the individual losses in a meaningful fashion.

### Squared loss: a popular loss function

The linear regression models we'll examine here use a loss function called **squared loss** (also known as **L2 loss**). The squared loss for a single example is as follows:

```
  = the square of the difference between the label and the prediction
  = (observation - prediction(x))2
  = (y - y')2
```

**Mean square error** (**MSE**) is the average squared loss per example over the whole dataset. To calculate MSE, sum up all the squared losses for individual examples and then divide by the number of examples:



$MSE=1/N∑(x,y)∈D(y−prediction(x))2$



where:

- 

  (x,y) is an example in which

  - x is the set of features (for example, chirps/minute, age, gender) that the model uses to make predictions.
  - y is the example's label (for example, temperature).

- prediction(x) is a function of the weights and bias in combination with the set of features x.

- D is a data set containing many labeled examples, which are (x,y) pairs.

- N is the number of examples in D.

Although MSE is commonly-used in machine learning, it is neither the only practical loss function nor the best loss function for all circumstances.





# Reducing Loss



To train a model, we need a good way to reduce the model’s loss. An iterative approach is one widely used method for reducing loss, and is as easy and efficient as walking down a hill.

**Learning Objectives** Discover how to train a model using an iterative approach.Understand full gradient descent and some variants, including:mini-batch gradient descent stochastic gradient descent Experiment with learning rate.



## How do we reduce loss?

- Hyperparameters are the configuration settings used to tune how the model is trained.

- Derivative of (y - y')2 with respect to the weights and biases tells us how loss changes for a given example

- - Simple to compute and convex

- So we repeatedly take small steps in the direction that minimizes loss

- - We call these **Gradient Steps** (But they're really negative Gradient Steps)
  - This strategy is called **Gradient Descent**



## Block Diagram of Gradient Descent



![The cycle of moving from features and labels to models and predictions.](https://developers.google.com/machine-learning/crash-course/images/GradientDescentDiagram.svg)

**Figure 1. An iterative approach to training a model.**





## SGD & Mini-Batch Gradient Descent

- Could compute gradient over entire data set on each step, but this turns out to be unnecessary

- Computing gradient on small data samples works well

- - On every step, get a new random sample

- **Stochastic Gradient Descent**: one example at a time

- **Mini-Batch Gradient Descent**: batches of 10-1000

- - Loss & gradients are averaged over the batch





# Reducing Loss: An Iterative Approach



Iterative learning might remind you of the ["Hot and Cold"](http://www.howcast.com/videos/258352-how-to-play-hot-and-cold/) kid's game for finding a hidden object like a thimble. In this game, the "hidden object" is the best possible model. You'll start with a wild guess ("The value of w1 is 0.") and wait for the system to tell you what the loss is. Then, you'll try another guess ("The value of w1 is 0.5.") and see what the loss is. Aah, you're getting warmer. Actually, if you play this game right, you'll usually be getting warmer. The real trick to the game is trying to find the best possible model as efficiently as possible.

The above figure suggests the iterative trial-and-error process that machine learning algorithms use to train a model:







We'll use this same iterative approach throughout Machine Learning Crash Course, detailing various complications, particularly within that stormy cloud labeled "Model (Prediction Function)." Iterative strategies are prevalent in machine learning, primarily because they scale so well to large data sets.

The "model" takes one or more features as input and returns one prediction (y') as output. To simplify, consider a model that takes one feature and returns one prediction:



​                                                  $y′=b+w1x1$

What initial values should we set for b and w1? For linear regression problems, it turns out that the starting values aren't important. We could pick random values, but we'll just take the following trivial values instead:

- b = 0
- w1 = 0

Suppose that the first feature value is 10. Plugging that feature value into the prediction function yields:

```
  y' = 0 + 0(10)
  y' = 0
```

The "Compute Loss" part of the diagram is the [loss function](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss) that the model will use. Suppose we use the squared loss function. The loss function takes in two input values:

- *y'*: The model's prediction for features *x*
- *y*: The correct label corresponding to features *x*.

At last, we've reached the "Compute parameter updates" part of the diagram. It is here that the machine learning system examines the value of the loss function and generates new values for b and w1. For now, just assume that this mysterious box devises new values and then the machine learning system re-evaluates all those features against all those labels, yielding a new value for the loss function, which yields new parameter values. And the learning continues iterating until the algorithm discovers the model parameters with the lowest possible loss. Usually, you iterate until overall loss stops changing or at least changes extremely slowly. When that happens, we say that the model has **converged**.



**Key Point:**

A Machine Learning model is trained by starting with an initial guess for the weights and bias and iteratively adjusting those guesses until learning the weights and bias with the lowest possible loss.



# Reducing Loss: Gradient Descent





The iterative approach diagram ([Figure 1](https://developers.google.com/machine-learning/crash-course/reducing-loss/an-iterative-approach#ml-block-diagram)) contained a green hand-wavy box entitled "Compute parameter updates." We'll now replace that algorithmic fairy dust with something more substantial.

Suppose we had the time and the computing resources to calculate the loss for all possible values of w1. For the kind of regression problems we've been examining, the resulting plot of loss vs. w1 will always be convex. In other words, the plot will always be bowl-shaped, kind of like this:

![A second point on the U-shaped curve, this one a little closer to the minimum point.](https://developers.google.com/machine-learning/crash-course/images/convex.svg)

**Figure 2. Regression problems yield convex loss vs weight plots.**

 

Convex problems have only one minimum; that is, only one place where the slope is exactly 0. That minimum is where the loss function converges.

Calculating the loss function for every conceivable value of w1 over the entire data set would be an inefficient way of finding the convergence point. Let's examine a better mechanism—very popular in machine learning—called **gradient descent**.

The first stage in gradient descent is to pick a starting value (a starting point) for w1. The starting point doesn't matter much; therefore, many algorithms simply set w1 to 0 or pick a random value. The following figure shows that we've picked a starting point slightly greater than 0:

![A second point on the U-shaped curve, this one a little closer to the minimum point.](https://developers.google.com/machine-learning/crash-course/images/GradientDescentStartingPoint.svg)

**Figure 3. A starting point for gradient descent.**

The gradient descent algorithm then calculates the gradient of the loss curve at the starting point. Here in Figure 3, the gradient of loss is equal to the [derivative](https://wikipedia.org/wiki/Differential_calculus#The_derivative) (slope) of the curve, and tells you which way is "warmer" or "colder." When there are multiple weights, the **gradient** is a vector of partial derivatives with respect to the weights.





Note that a gradient is a vector, so it has both of the following characteristics:

- a direction
- a magnitude

The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.

![A second point on the U-shaped curve, this one a little closer to the minimum point.](https://developers.google.com/machine-learning/crash-course/images/GradientDescentNegativeGradient.svg)

**Figure 4. Gradient descent relies on negative gradients.**

To determine the next point along the loss function curve, the gradient descent algorithm adds some fraction of the gradient's magnitude to the starting point as shown in the following figure:

![A second point on the U-shaped curve, this one a little closer to the minimum point.](https://developers.google.com/machine-learning/crash-course/images/GradientDescentGradientStep.svg)

**Figure 5. A gradient step moves us to the next point on the loss curve.**

The gradient descent then repeats this process, edging ever closer to the minimum.

**Note:** When performing gradient descent, we generalize the above process to tune all the model parameters *simultaneously*. For example, to find the optimal values of both w1 and the bias b, we calculate the gradients with respect to both w1 and b. Next, we modify the values of w1 and b based on their respective gradients. Then we repeat these steps until we reach minimum loss.





# Reducing Loss: Learning Rate



As noted, the gradient vector has both a direction and a magnitude. Gradient descent algorithms multiply the gradient by a scalar known as the **learning rate** (also sometimes called **step size**) to determine the next point. For example, if the gradient magnitude is 2.5 and the learning rate is 0.01, then the gradient descent algorithm will pick the next point 0.025 away from the previous point.

**Hyperparameters** are the knobs that programmers tweak in machine learning algorithms. Most machine learning programmers spend a fair amount of time tuning the learning rate. If you pick a learning rate that is too small, learning will take too long:

![Same U-shaped curve. Lots of points are very close to each other and their trail is making extremely slow progress towards the bottom of the U.](https://developers.google.com/machine-learning/crash-course/images/LearningRateTooSmall.svg)

**Figure 6. Learning rate is too small.**

Conversely, if you specify a learning rate that is too large, the next point will perpetually bounce haphazardly across the bottom of the well like a quantum mechanics experiment gone horribly wrong:

![Same U-shaped curve. This one contains very few points. The trail of points jumps clean across the bottom of the U and then jumps back over again.](https://developers.google.com/machine-learning/crash-course/images/LearningRateTooLarge.svg)

**Figure 7. Learning rate is too large.**

There's a [Goldilocks](https://wikipedia.org/wiki/Goldilocks_principle) learning rate for every regression problem. The Goldilocks value is related to how flat the loss function is. If you know the gradient of the loss function is small then you can safely try a larger learning rate, which compensates for the small gradient and results in a larger step size.

![Same U-shaped curve. The trail of points gets to the minimum point in about eight steps.](https://developers.google.com/machine-learning/crash-course/images/LearningRateJustRight.svg)

**Figure 8. Learning rate is just right.**



#### ideal learning rate.

The ideal learning rate in one-dimension is $1f(x)″$ (the inverse of the second derivative of f(x) at x).

The ideal learning rate for 2 or more dimensions is the inverse of the [Hessian](https://wikipedia.org/wiki/Hessian_matrix) (matrix of second partial derivatives).

The story for general convex functions is more complex.





# Reducing Loss: Stochastic Gradient Descent



In gradient descent, a **batch** is the total number of examples you use to calculate the gradient in a single iteration. So far, we've assumed that the batch has been the entire data set. When working at Google scale, data sets often contain billions or even hundreds of billions of examples. Furthermore, Google data sets often contain huge numbers of features. Consequently, a batch can be enormous. A very large batch may cause even a single iteration to take a very long time to compute.

A large data set with randomly sampled examples probably contains redundant data. In fact, redundancy becomes more likely as the batch size grows. Some redundancy can be useful to smooth out noisy gradients, but enormous batches tend not to carry much more predictive value than large batches.

What if we could get the right gradient *on average* for much less computation? By choosing examples at random from our data set, we could estimate (albeit, noisily) a big average from a much smaller one.**Stochastic gradient descent** (**SGD**) takes this idea to the extreme--it uses only a single example (a batch size of 1) per iteration. Given enough iterations, SGD works but is very noisy. The term "stochastic" indicates that the one example comprising each batch is chosen at random.

**Mini-batch stochastic gradient descent** (**mini-batch SGD**) is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.

To simplify the explanation, we focused on gradient descent for a single feature. Rest assured that gradient descent also works on feature sets that contain multiple features.





# First Steps with TensorFlow



**Learning Objectives**

Learn how to create and modify tensors in TensorFlow.

Learn the basics of pandas.

Develop linear regression code with one of TensorFlow's high-level APIs.

Experiment with learning rate.







## TensorFlow API Hierarchy

![Hierarchy of TensorFlow toolkits. Estimator API is at the top.](https://developers.google.com/machine-learning/crash-course/images/TFHierarchy.svg)





# First Steps with TensorFlow: Toolkit





Tensorflow is a computational framework for building machine learning models. TensorFlow provides a variety of different toolkits that allow you to construct models at your preferred level of abstraction. You can use lower-level APIs to build models by defining a series of mathematical operations. Alternatively, you can use higher-level APIs (like `tf.estimator`) to specify predefined architectures, such as linear regressors or neural networks.

The following figure shows the current hierarchy of TensorFlow toolkits:

![Hierarchy of TensorFlow toolkits. Estimator API is at the top.](https://developers.google.com/machine-learning/crash-course/images/TFHierarchy.svg)

**Figure 1. TensorFlow toolkit hierarchy.**

The following table summarizes the purposes of the different layers:

| Toolkit(s)                     | Description                            |
| :----------------------------- | :------------------------------------- |
| Estimator (tf.estimator)       | High-level, OOP API.                   |
| tf.layers/tf.losses/tf.metrics | Libraries for common model components. |
| TensorFlow                     | Lower-level APIs                       |

TensorFlow consists of the following two components:

- a [graph protocol buffer](https://www.tensorflow.org/extend/tool_developers/#protocol_buffers)
- a runtime that executes the (distributed) graph

These two components are analogous to Python code and the Python interpreter. Just as the Python interpreter is implemented on multiple hardware platforms to run Python code, TensorFlow can run the graph on multiple hardware platforms, including CPU, GPU, and [TPU](https://wikipedia.org/wiki/Tensor_processing_unit).

Which API(s) should you use? You should use the highest level of abstraction that solves the problem. The higher levels of abstraction are easier to use, but are also (by design) less flexible. We recommend you start with the highest-level API first and get everything working. If you need additional flexibility for some special modeling concerns, move one level lower. Note that each level is built using the APIs in lower levels, so dropping down the hierarchy should be reasonably straightforward.

## tf.estimator API

We'll use tf.estimator for the majority of exercises in Machine Learning Crash Course. Everything you'll do in the exercises could have been done in lower-level (raw) TensorFlow, but using tf.estimator dramatically lowers the number of lines of code.

tf.estimator is compatible with the scikit-learn API. [Scikit-learn](http://scikit-learn.org/) is an extremely popular open-source ML library in Python, with over 100k users, including many at Google.

Very broadly speaking, here's the pseudocode for a linear classification program implemented in tf.estimator:

```python
import tensorflow as tf

# Set up a linear classifier.
classifier = tf.estimator.LinearClassifier(feature_columns)

# Train the model on some example data.
classifier.train(input_fn=train_input_fn, steps=2000)

# Use it to predict.
predictions = classifier.predict(input_fn=predict_input_fn)
```





# First Steps with TensorFlow: Programming Exercises







### Is There a Standard Heuristic for Model Tuning?

This is a commonly asked question. The short answer is that the effects of different hyperparameters are data dependent. So there are no hard-and-fast rules; you'll need to test on your data.

That said, here are a few rules of thumb that may help guide you:

- Training error should steadily decrease, steeply at first, and should eventually plateau as training converges.
- If the training has not converged, try running it for longer.
- If the training error decreases too slowly, increasing the learning rate may help it decrease faster.
  - But sometimes the exact opposite may happen if the learning rate is too high.
- If the training error varies wildly, try decreasing the learning rate.
  - Lower learning rate plus larger number of steps or larger batch size is often a good combination.
- Very small batch sizes can also cause instability. First try larger values like 100 or 1000, and decrease until you see degradation.

Again, never go strictly by these rules of thumb, because the effects are data dependent. Always experiment and verify.





## Common hyperparameters in Machine Learning Crash Course exercises

Many of the coding exercises contain the following hyperparameters:

- **steps**, which is the total number of training iterations. One step calculates the loss from *one batch* and uses that value to modify the model's weights *once*.
- **batch size**, which is the number of examples (chosen at random) for a single step. For example, the batch size for SGD is 1.

The following formula applies:



$total   number  of trained examples=batchsize∗steps$



## A convenience variable in Machine Learning Crash Course exercises

The following convenience variable appears in several exercises:

- **periods**, which controls the granularity of reporting. For example, if `periods` is set to 7 and `steps` is set to 70, then the exercise will output the loss value every 10 steps (or 7 times). Unlike hyperparameters, we don't expect you to modify the value of `periods`. Note that modifying `periods` does not alter what your model learns.

The following formula applies:



$numberoftrainingexamplesineachperiod=batchsize∗stepsperiods$







# Generalization



**Generalization** refers to your model's ability to adapt properly to new, previously unseen data, drawn from the same distribution as the one used to create the model.





## The Big Picture

![Cycle of model, prediction, sample, discovering true distribution, more sampling](https://developers.google.com/machine-learning/crash-course/images/BigPicture.svg)



- Goal: predict well on new data drawn from (hidden) true distribution.

- Problem: we don't see the truth.

- - We only get to sample from it.

- If model h fits our current sample well, how can we trust it will predict well on other new samples?





## How Do We Know If Our Model Is Good?

- Theoretically:

- - Interesting field: generalization theory
  - Based on ideas of measuring model simplicity / complexity

- Intuition: formalization of Ockham's Razor principle

- - The less complex a model is, the more likely that a good empirical result is not just due to the peculiarities of our sample

Empirically:

- Asking: will our model do well on a new sample of data?

- Evaluate: get a new sample of data-call it the test set

- Good performance on the test set is a useful indicator of good performance on the new data in general:

- - If the test set is large enough
  - If we don't cheat by using the test set over and over

- 

## The ML Fine Print

Three basic assumptions in all of the above:

1. We draw examples **independently and identically (i.i.d.)** at random from the distribution
2. The distribution is **stationary**: It doesn't change over time
3. We always pull from the **same distribution**: Including training, validation, and test sets





