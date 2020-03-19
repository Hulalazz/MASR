# Decision Tree and Polynomials Regression

For simplicity, we suppose that all variables are numerical and the decision tree can be expressed in a polynomials form:
$$T(x)=\sum_{\ell}^{|T|} c_{\ell}\prod_{i\in P(\ell)}\sigma(\pm(x_i)-\tau_i)$$
where the index $\ell$ is referrred to the leaf node;
the constant $|T|$ is the total number of leaf nodes;
the set $P(\ell)$ is the node in the path from the root to the leaf node $\ell$;
the function $\sigma(\cdot)$ is the [unit step function];
the input $x=(x_1, x_2,\cdots, x_d)\in\mathbb{R}^d$ and $x_i\in\mathbb{R}$ is the component of the input $x$;
the constant $c_{\ell}$ is the constant associated with the leaf node.

The product of step fuctions is the [indicator functions][IndicatorFunctions].
For example, $T(x)=a\sigma(x_1 - 1) + b\sigma(1 - x_1)$, is the decision tree with only the root node and two leaf nodes where $x-1>1$ it returns $a$; otherwise, it return $b$.


For categorical variables $z=(z_1, z_1,\dots, z_n)$, each component $z_i$ for $i=1,2,\cdots, n$ takes finite categorical values.
Embedding techniques will encode these categorical variables into digital codes. In another word, they map each categorical values into unique numerical fearure.
For example, the hashing function  can map every string into unique codes.
For categoical variable, the decision trees will perform the `equalsto(==)` test, i.e., if the variable $x_i$ is equal to a given value, it returns 1; otherwise it returns 0.
Based on this obsrvation, we can apply [Interpolation Methods](http://www.gisresources.com/types-interpolation-methods_3/).
Suppose the categotical variable $z_1$ embedded in $\{a_1, a_2, a_3,\cdots, a_n\}$, the [Lagrange Interpolation Formula](https://byjus.com/lagrange-interpolation-formula/) gives $\sum_{i=1}^{n}\prod_{j\not=i}\frac{(z_1-a_j)}{(a_i -a_j)}$, where $\prod_{j\not=i}\frac{(z_1-a_j)}{(a_i -a_j)}$ equals to 1 if $z_1=a_i$ otherwise it is equal to 0.


In [MARS], the recursive partitioning regression, as binary regression tree,  is  viewed in a more conventional light as a stepwise regression procedure:
$$f_M(x)=\sum_{m=1}^{M}a_m\prod_{k=1}^{K_m}H[s_{km}(x_{v(k, m) - t_{km}})]$$
where $H(\cdot)$ is the [unit step function].
The quantity $K_m$, is the number of splits that gave rise to basis function.
The quantities $s_{km}$ take on values k1and indicate the (right/left) sense of the associated step function.
The $v(k, m )$ label the predictor variables and 
the $t_{km}$, represent values on the corresponding variables.
The internal nodes of the binary tree represent the step functions and the terminal nodes represent the final basis functions.

_____

[Delta Function](https://mathworld.wolfram.com/DeltaFunction.html) is the derivative of step function, an example of [generalized function](https://mathworld.wolfram.com/GeneralizedFunction.html);
the unit step function is the derivattive of the [Ramp Function](https://mathworld.wolfram.com/RampFunction.html)  defined by
$$R(x)=x\sigma(x)=\max(0,x)=ReLU(x)=(x)^{+}.$$

[Kronecker Delta](https://mathworld.wolfram.com/KroneckerDelta.html) is defined as 
$$\delta_{ij}=\begin{cases}1, \text{if $i=j$}\\
0, \text{otherwise}\end{cases}.$$

The [indicator functions][IndicatorFunctions] is defined as
$$\mathbb{1}_{ \{condition\} }=\begin{cases}1, &\text{if condition is hold }\\
0, &\text{otherwise}.\end{cases}$$

Given a subset $A$ of a larger set, the `characteristic function` $\chi_A$, sometimes also called the **indicator function**, is the function defined to be identically one on $A$, and is zero elsewhere[CharacteristicFunction]. 

A simple function is a finite sum $sum_(i)a_i \chi_(A_i)$, where the functions $\chi_(A_i)$ are characteristic functions on a set $A$. Another description of a simple function is a function that takes on finitely many values in its range[SimpleFunction].


****

The foundamental limitaion of decision tree is: 
- its lack of `continuity`, 
- lack `smooth [decision boundary]`,
- inability to provide good approximations to certain classes of simple often-occurring functions.

The decision tree is simple fucntion essentially.
Based on above observation, decision tree is considered  to project the raw data into distinct decisin region in an adaptive approach and select the mode of within-node samples groudtruth as their `label`.


We can replace the unit step function with truncated power series as in [MARS] to tackle the discontinuty to invent continous models with continuous derivatives.
[MARS] is an extention of recursive partitioning regression:
$$f_M(x)=\sum_{m=1}^{M}a_m\prod_{k=1}^{K_m}[s_{km}(x_{v(k, m) - t_{km}})]^{+}.$$
Note that the ramp function $(x)^{+}=\max(0,x)=x\cdot H(x)$ so we can re-express the MARS as 
$$f_M(x)=\sum_{m=1}^{M}a_m(x)\prod_{k=1}^{K_m}H[s_{km}(x_{v(k, m) - t_{km}})]$$
where $a_m(x)=\prod_{k=1}^{K_m}(x_{v(k, m) - t_{km}})$.
[MARS] can approximate the polynomials directly. 
For example, the identity function is $f_M(x)=\sum_{m=1}^{M}x\prod_{k=1}^{K_m}[s_{km}(x_{v(k, m) - t_{km}})]^{+}$.


The decision boundary of both methods is `axis-aligned` and therefor not smooth.
In fact, the product $\prod_{k=1}^{K_m}H[s_{km}(x_{v(k, m) - t_{km}})]$ is the characteristic function of axis-aligned set $S=\{s_{km}(x_{v(k, m) - t_{km}})>0,k=1,2,\cdots, K-m\}$.
This is becaue of the relation of ramp functriona and unit step function.


This can be solved by data preprocessing and feature transformation such as kernel technique.
Or we can modify the unit step function. 
For example, we can use the multiple linear combination of features to substitute the univariate function: $\max\{0, \left<w,x\right>\}$.
In this case the [decision boundary] is piece-wise linear.
[B-spline] and [Bézier Curve] can be used to describe the decision boundary.
Observe that $xH(x)=\max\{0,x\}$, $\max(0, \operatorname{sgn}x^2)=x^2H(x)$ 
where $H(x)$ is the unit step function while $\max(0, x^2+x^3)\not= (x^2+x^3)H(x)$.

We will generalize the decision boundary into [decision manifold](http://www.ifs.tuwien.ac.at/~lidy/pub/poe_lidy_wsom07.pdf).



_____

In [soft decision trees](http://www.cs.cornell.edu/~oirsoy/softtree.html), the unit step function $\sigma(\cdot)$ is repalced by the smooth sgmoid or logstic function $\sigma(x)=\frac{1}{1+\exp(-x)}$.

[Learning pairwise image similarities for multi-classification using Kernel Regression Trees](http://www.brunel.ac.uk/~csstyyl/papers/pr2012.pdf)

[Adaptive Neural Trees](https://github.com/rtanno21609/AdaptiveNeuralTrees) gain the complementary benefits of neural networks and decision trees.

[Competitive Neural Trees for Pattern Classification](https://www.ais.uni-bonn.de/behnke/papers/tnn98.pdf) contains m-ary nodes and
grows during learning by using inheritance to initialize new
nodes.



******

Many mulptile adaptive regresion methods are specializations of a general multivariate
regression algorithm that builds `hierarchical models` using a set of basis
functions and `stepwise selection`:
$$f_M(x,\theta)=\sum_{m=1}^{M}\theta_m B_m(x)$$
for $x\in\mathbb{R}^n$.

Let us compare  hierarchical models including
decision tree, multiple adaptive regression spline and [Recursive partitioning regression](https://projecteuclid.org/download/pdf_1/euclid.aos/1176347963).


Decision Tree| MARS| Multivariate adaptive polynomial synthesis (MAPS)
-------------|---------|-------
Discontinuity|Smoothness|----
Unit step function|ReLU|



- https://www.salford-systems.com/products/mars
- [MARS](http://www.stat.yale.edu/~lc436/08Spring665/Mars_Friedman_91.pdf)
- [Discussion](http://www.stat.yale.edu/~arb4/publications_files/DiscussionMultivariateAdaptiveRegressionSplines.pdf)

*****

[unit step function]: https://mathworld.wolfram.com/HeavisideStepFunction.html
[IndicatorFunctions]: https://www.statlect.com/fundamentals-of-probability/indicator-functions
[CharacteristicFunction]:https://mathworld.wolfram.com/CharacteristicFunction.html
[SimpleFunction]: https://mathworld.wolfram.com/SimpleFunction.html
[MARS]: https://projecteuclid.org/download/pdf_1/euclid.aos/1176347963
[decision boundary]: https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/PR_simp/bndrys.htm
[B-spline]: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html
[Bézier Curve]: https://mathworld.wolfram.com/BezierCurve.html