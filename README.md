# polynet

This application is a neural network based tool that is designed to find polynomial which fits some collection of points the best.

## Network's architecture

NN input is single x value (per example). Its output is a vector of coefficients of the approximated polynomial. 

NN used in this project does not need to be more than 1 layer deep (0 hidden layers), because it will overfit anyway. This single layer has linear activation. It contains as many units as POLYNOMIAL_DEGREE + 1 value used as an input to the application during training. 

## Usage

#### Training

`python polynomial.py train  POLYNOMIAL_DEGREE PATH_TO_CSV`

- POLYNOMIAL_DEGREE - upper bound on the degree of a polynomial which will be approximated.
- PATH_TO_CSV - path to the .csv file which contains dataset of points. Application will try to fit polynomial to these points.

#### Inference

`python polynomial.py estimate X`

- X - float value of x for which y will be estimated using previously trained neural network.

### 

## Requirements

- Python 2.7.14 (other versions may work too, but it was not tested),
- numpy 1.14.2 (same as above)
