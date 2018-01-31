function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h0 = sigmoid(X * theta);
vct_J = -y .* log(h0) - (1 - y) .* log(1 - h0);

theta_except_first = (theta(2 : end , :));
J = sum(vct_J) / m + (sum( theta_except_first.^ 2)) * lambda / (2 * m);

h0_diff = h0 - y;
for iter = 1 : size(theta, 1),
  vct_grad = h0_diff .* X( : , iter);
  grad(iter) = sum(vct_grad) / m;
  if (iter != 1)
    grad(iter) += lambda * theta( iter , : ) / m;
  endif
end;

% =============================================================

end
