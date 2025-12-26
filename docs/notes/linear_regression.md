## Linear Regression

Linear regression and specifically ordinary least squares may be considered from many, many angles: statistics, linear algebra, calculus, optimization. Let’s discuss each and how they fit together.

### Statistics of Regression: Estimation

Consider a joint distribution of random variables $(X, Y)$. We think that there should be some relationship between them. Suppose that we have $n$ different samples from this distribution, i.e. $n$ different independent pairs $(X_i, Y_i)$. Suppose then that, when we have access to only the observation of $X$ but not $Y$, we want to predict what $Y$ will be. Examples could include when $Y$ is something that will occur in the <i>future</i>, such as a patient's health outcome of a surgery. We would have access to the patient's current health status and metrics, such as age, BMI, blood sugar, and blood pressure.

In statistics, our question would be: how can we best predict the $Y$ if given the $X$? A natural approach would be to make a best guess of what $Y$ would be on <i>average</i> for a specific $X$. This would amount to estimation the <i>regression function</i>.

<div class="callout definition"><span class="label">Definition: Regression Function</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
Let $\mathbb P_{X, Y}$ be a distribution on $\mathbb R^p \times \mathbb R$ where $X \in \mathbb R^p$ are the <i>predictors</i>, or <i>independent variables</i>, or <i>explanatory variables</i>, and $Y$ is the <i>response</i> or the <i>dependent</i> variable.

The <strong><i>regression function</i></strong> is defined as
$$f(x) \triangleq \mathbb E[Y | X = x].$$
for the realization $x$ of $X$.
</div>

What, then, does <i>linear</i> regression mean? 

It turns out that, in many scenarios, this regression function $\mathbb E[Y | X = x]$ is or can be well approximated by a linear function of $X$. And even if it isn't, we might make this assumption for simplicity. This looks like

$$f(x) = x^\top \beta$$
for $\beta \in \mathbb R^{p}$ which&mdash;if an intercept estimator is wanted&mdash;contains a column of ones to represent the intercept. Given this, we may also write, for each pair $(X_i, Y_i)$,
$$Y_i = X_i^\top \beta + \epsilon_i,$$
where $\epsilon$ is some error. We leave the discussion of the assumptions on $\epsilon$ for later.

For now, then, our objective is to estimate this $\beta$. There are a couple different ways we can do this. We'll describe two statistical perspectives that both lead to the OLS estimator.

#### Decision Theory: Square Error Loss

For example, through the framework of <i>decision theory</i>, we should define a loss function and aim to minimize it. From now on, for the rest of this section, we shall assume that the true regression function $f(x)$ is of the form $f(x) = x^\top \beta$ for some $\beta \in \mathbb R^p$.

<div class="callout definition"><span class="label">Definition: Loss Function</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
Let $\beta \in \mathbb R^p$. For an estimator $\hat\beta$ of $\beta$, the <strong><i>loss function</i></strong> is a function $\ell(\hat\beta, \beta)$ such that $\ell(\hat\beta, \beta) \geq 0$ and $\ell(\beta, \beta) = 0$.
</div>

Using the decision theoretic framework, we must then select a specific loss function to minimize. A common choice is the <i>square error loss</i> or the <i>L2 norm loss</i>, defined as $\ell(\hat\beta, \beta) = \Vert \hat\beta - \beta \Vert^2_2$. Minimizing this function is where ordinary least squares comes from.

#### Maximum Likelihood: Gaussian Errors

Another natural connection of OLS is with maximum likelihood estimation for when the errors are gaussian. Specifically, assume that
$$Y_i = X_i^\top \beta + \epsilon_i,$$
where $\epsilon \sim \mathcal N(0, \sigma^2)$ for some variance $\sigma^2 > 0$.

When we have $n$ samples, i.e. $(X_i, Y_i)$, and we want to estimate $\beta$, using the maximum likelihood approach, we would aim to maximize the log likelihood:
$$\hat\beta = \max_{\beta \in \mathbb R^p} \mathcal L(\beta) = \log\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^n - \sum_{i = 1}^n \left(\frac{1}{2\sigma^2}(y_i - x_i^\top \beta)\right).$$
We see that this is equivalent to minimizing the least squares objective.

### 