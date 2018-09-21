function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    %OPTIMIZED SOLUTION
    theta = theta - ((alpha/m)*(((X*theta)-y)'*X)');

%    for i=1:size(X,2)
%        grad = theta(i)-alpha*(1/m)*sum((X*theta-y).*X(:,i));
%        theta(i) = grad;
%    end
    % ============================================================
    printf('Iteration nr %d. Gradients are: %f, %f, %f\n',iter,theta);

    % Save the cost J in every iteration    

    cost_func = computeCost(X, y, theta);
    printf('Cost function after gradient updates is: %f \n',cost_func);
    
    J_history(iter) = cost_func;

end
