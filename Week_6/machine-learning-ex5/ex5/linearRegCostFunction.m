function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Returning cost function
yhat = X * theta
thetaTemp = theta(2:end)
tempCost = ((yhat - y) .^ 2)
tempSum = sum(tempCost) / (2*m)
jReg = ((lambda/(2*m)) * (sum(thetaTemp .^ 2)))
J = tempSum + jReg

% Returning regression gradient
grad = ((X' * (yhat - y)) / m) + ((lambda/m) * [0;thetaTemp])
grad = grad(:)












% =========================================================================

grad = grad(:);

end
