function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %
    grad1 = theta(1)-alpha*(1/m)*sum((X*theta-y).*X(:,1));
    grad2 = theta(2)-alpha*(1/m)*sum((X*theta-y).*X(:,2));
    theta = [grad1;grad2];
    
    printf('Gradient 1 is: %f | gradient 2 is: %f \n', theta(1),theta(2));

    % ============================================================

    % Save the cost J in every iteration    
    cost_func = computeCost(X, y, theta);
    printf('Cost function after gradient updtaes is: %f \n',cost_func);
    J_history(iter) = cost_func;

end

end
