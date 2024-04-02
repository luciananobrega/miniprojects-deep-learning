# Polynomial regression

## Dataset Description

The dataset used in this work is based on two random variables, $X$ and $Y$. The $X$ values are randomly sampled from a uniform distribution within the range $(0, 1)$, and the $Y$ values are related to $X$ according to the following equation:

$$Y = \cos(2\pi X) + Z,$$

where $Z$ represents a Gaussian random variable with zero mean and a variance of $\sigma^2$, statistically independent of $X$.

## Data Generation
Based on this equation, we obtain the dataset $\{(x_i, y_i): i=1,2,\dots, N\}$ for any amount of data $N$ and any variance $\sigma^2$.

## Data Estimation
Considering that the relation between $X$ and $Y$ is unknown, we employ polynomial regressions with and without regularization for the data estimation.

Assuming $d$ as the polynomial degree (i.e., the model complexity) and $a_i$ as the model coefficients, we can estimate $Y$ as $\hat{Y}$, employing a polynomial regression model:
$$\hat{Y} = a_0 + a_1X + a_2 X^2 + \dots + a_d X^d.$$
With this, we can estimate the values of $y_i$ based on the value of $x_i$. The estimated value of $y_i$ is defined as $\hat{y}_i$.

## System Parameters

| Variable      | Description               | Value               |
| ------------- | ------------------------- | ------------------- |
| $N$           | In-sample size            | $\{2,5,10,20,50,100,200\}$ |
| $d$           | Model polynomial degree   | $\{1,2,4,8,16,32,64\}$ |
| $\sigma$      | Noise variance            | $\{0.05, 0.2\}$     |
| $N_{out}$     | Out-of-sample size        | $2000$              |
| $M$           | Number of trials          | $50$                |
| $\lambda$     | Learning rate             | $0.001$             |
| $\lambda_{Reg}$ | Weight decay            | $0.1$               |
| $e$           | Number of epochs          | $5000$              |

## Regularization Improvement on $E_{out}$ for $d=16$

Table below demonstrates the improvement on $E_{out}$ for $d=16$, where **NonReg** and **Reg** represent the non-regularized and regularized case, respectively.

| **$N$** | $\sigma$ | **$E_{out}$ NonReg** | **$E_{out}$ Reg** | **Improvement** |
| ------- | -------- | --------------------- | ------------------ | --------------- |
| 2       | 0.05     | 1.277062              | 0.736007           | 42.37\%         |
|         | 0.2      | 1.364742              | 0.82926            | 39.24\%         |
| 5       | 0.05     | 0.697354              | 0.455785           | 34.64\%         |
|         | 0.2      | 0.959961              | 0.599685           | 37.53\%         |
| 10      | 0.05     | 0.493634              | 0.393032           | 20.38\%         |
|         | 0.2      | 0.501589              | 0.4241             | 15.45\%         |
| 20      | 0.05     | 0.39183               | 0.350458           | 10.56\%         |
|         | 0.2      | 0.43333               | 0.393277           | 9.24\%          |
| 50      | 0.05     | 0.321905              | 0.326724           | -1.50\%         |
|         | 0.2      | 0.384219              | 0.376926           | 1.90\%          |
| 100     | 0.05     | 0.323126              | 0.328613           | -1.70\%         |
|         | 0.2      | 0.354502              | 0.362274           | -2.19\%         |
| 200     | 0.05     | 0.314602              | 0.3242             | -3.05\%         |
|         | 0.2      | 0.354029              | 0.362077           | -2.27\%         |

## Regularization Improvement on $E_{out}$ for $d=64$

Table below demonstrates the improvement on $E_{out}$ for $d=64$, where **NonReg** and **Reg** represent the non-regularized and regularized case, respectively.

| **$N$** | $\sigma$ | **$E_{out}$ NonReg** | **$E_{out}$ Reg** | **Improvement** |
| ------- | -------- | --------------------- | ------------------ | --------------- |
| 2       | 0.05     | 10.35228              | 3.904016           | 62.29\%         |
|         | 0.2      | 11.99375              | 4.643942           | 61.28\%         |
| 5       | 0.05     | 7.296974              | 2.821608           | 61.33\%         |
|         | 0.2      | 6.904801              | 2.707602           | 60.79\%         |
| 10      | 0.05     | 5.646942              | 2.26447            | 59.90\%         |
|         | 0.2      | 5.263499              | 2.089125           | 60.31\%         |
| 20      | 0.05     | 2.671688              | 1.182629           | 55.73\%         |
|         | 0.2      | 3.596994              | 1.561084           | 56.60\%         |
| 50      | 0.05     | 1.071691              | 0.635487           | 40.70\%         |
|         | 0.2      | 1.292521              | 0.729691           | 43.55\%         |
| 100     | 0.05     | 0.522184              | 0.424192           | 18.77\%         |
|         | 0.2      | 0.576945              | 0.46647            | 19.15\%         |
| 200     | 0.05     | 0.428846              | 0.385108           | 10.20\%         |
|         | 0.2      | 0.441148              | 0.412385           | 6.52\%          |