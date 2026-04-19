from sklearn.datasets import load_diabetes
import pandas as pd

# load in diabetes dataset
X, y =load_diabetes(return_X_y=True, as_frame=True)

"""
X = input (feature-ök)

Ez a beteg adatai:

age
sex
bmi
blood pressure
s1–s6 (labor értékek)

y: target -> Ez lesz a modell bemenete

y = target (amit tanítunk)
egy szám (pl. 151, 75, 206...)
"""
# Test feautres
print(X.head())

"""
        age       sex       bmi        bp  ...        s3        s4        s5        s6
0  0.038076  0.050680  0.061696  0.021872  ... -0.043401 -0.002592  0.019907 -0.017646
1 -0.001882 -0.044642 -0.051474 -0.026328  ...  0.074412 -0.039493 -0.068332 -0.092204
2  0.085299  0.050680  0.044451 -0.005670  ... -0.032356 -0.002592  0.002861 -0.025930
3 -0.089063 -0.044642 -0.011595 -0.036656  ... -0.036038  0.034309  0.022688 -0.009362
4  0.005383 -0.044642 -0.036385  0.021872  ...  0.008142 -0.002592 -0.031988 -0.046641
"""

# Get description of dataset
print(load_diabetes()["DESCR"])

"""
Diabetes dataset
----------------

Ten baseline variables, age, sex, body mass index, average blood
pressure, and six blood serum measurements were obtained for each of n =
442 diabetes patients, as well as the response of interest, a
quantitative measure of disease progression one year after baseline.

**Data Set Characteristics:**

:Number of Instances: 442

:Number of Attributes: First 10 columns are numeric predictive values

:Target: Column 11 is a quantitative measure of disease progression one year after baseline

:Attribute Information:
    - age     age in years
    - sex
    - bmi     body mass index
    - bp      average blood pressure
    - s1      tc, total serum cholesterol
    - s2      ldl, low-density lipoproteins
    - s3      hdl, high-density lipoproteins
    - s4      tch, total cholesterol / HDL
    - s5      ltg, possibly log of serum triglycerides level
    - s6      glu, blood sugar level

Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times the square root of `n_samples` (i.e. the sum of squares of each column totals 1).

Source URL:
https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

For more information see:
Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) "Least Angle Regression," Annals of Statistics (with discussion), 407-499.
(https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)
"""

# collectall the variables
all_variables = pd.concat([X,y], axis =1)
print(all_variables.head())

"""
  age       sex       bmi        bp  ...        s4        s5        s6  target
0  0.038076  0.050680  0.061696  0.021872  ... -0.002592  0.019907 -0.017646   151.0
1 -0.001882 -0.044642 -0.051474 -0.026328  ... -0.039493 -0.068332 -0.092204    75.0
2  0.085299  0.050680  0.044451 -0.005670  ... -0.002592  0.002861 -0.025930   141.0
3 -0.089063 -0.044642 -0.011595 -0.036656  ...  0.034309  0.022688 -0.009362   206.0
4  0.005383 -0.044642 -0.036385  0.021872  ... -0.002592 -0.031988 -0.046641   135.0
"""

print(all_variables.describe())

"""
                age           sex  ...            s6      target
count  4.420000e+02  4.420000e+02  ...  4.420000e+02  442.000000
mean  -2.511817e-19  1.230790e-17  ...  1.130318e-17  152.133484
std    4.761905e-02  4.761905e-02  ...  4.761905e-02   77.093005
min   -1.072256e-01 -4.464164e-02  ... -1.377672e-01   25.000000
25%   -3.729927e-02 -4.464164e-02  ... -3.317903e-02   87.000000
50%    5.383060e-03 -4.464164e-02  ... -1.077698e-03  140.500000
75%    3.807591e-02  5.068012e-02  ...  2.791705e-02  211.500000
max    1.107267e-01  5.068012e-02  ...  1.356118e-01  346.000000
"""