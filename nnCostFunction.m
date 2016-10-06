function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

% hidden et input layers car Theta1 contient les weights du passage entre ces 2 layers
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
% num_labels car = nb_output
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

X = [ones(m,1) X];
sum1 = 0;
yy = zeros(num_labels, 1);

a2 = zeros(m, hidden_layer_size);
h = zeros(m, num_labels);

%z3 = theta2*a2;
%h = g(z3);
a2 = sigmoid(X*Theta1');
a2 = [ones(size(a2,1),1) a2];
h = sigmoid(a2*Theta2');

% sum1 donnera une matrice 5000x1 avec à chq ligne les h de chq exemple. On somme sum1 pour donner le cout ensuite.
for i = 1:m
	% on déclare à 1 la i-ème case de y
	yy(y(i)) = 1;
	% on prend h ligne par ligne (1x10) qu'on multiplie par yy[10x1]. On obtient donc 5000 fois (boucle for) un élément 1x1 à chq tour de boucle
	% et on feed ça à sum1.
	sum1 = sum1 + (log(h(i,:))*(-yy)-log(1-h(i,:))*(1-yy));
	yy(y(i)) = 0;
end;

%J = (1/m)*sum(sumj) + (lambda/(2*m))*sum(theta(2:size(theta,1)).^2);

% ici, on somme la "somme" des 5000 h
J = (1/m)*sum(sum1);

% Theta1 : on veut d'abord la somme des carrés des éléments de l'input unit (1:400) puis des 25 éléments obtenus
% Theta2 : on veut la somme des carrés des 25 éléments du hidden unit puis celle des 10 éléments obtenus 
J = J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2, 2))+sum(sum(Theta2(:,2:end).^2, 2)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%aa2 = zeros(1, hidden_layer_size);
%hh = zeros(1, num_labels);
delta3 = zeros(m, num_labels);
yyy = zeros(1, num_labels);
acc_grad1 = zeros(size(Theta1));
acc_grad2 = zeros(size(Theta2));  

%for t = 1:m
	%aa1 = X(t, :);
	%aa2 = sigmoid(aa1*Theta1');
	%aa2 = [1 aa2];
	%hh = sigmoid(aa2*Theta2');
	%disp(size(hh));
	for i = 1:m
		yyy(y(i)) = 1;
		delta3(i, :) = h(i,:) - yyy;%5000x10
		yyy(y(i)) = 0;
	end;
	z2 = X*Theta1';%5000x25
	%disp(size(delta3));
	%z2 = aa1*Theta1';
	%z2 = [1 z2];
	delta2 = delta3*Theta2(:, 2:end).*sigmoidGradient(z2);%5000x25
	a1 = X;
	acc_grad1 = delta2'*a1; %25x401
	acc_grad2 = delta3'*a2; %10x26
%end;

	temp1 = Theta1;
	temp1(:, 1) = 0;
	temp2 = Theta2;
	temp2(:, 1) = 0;
	
	Theta1_grad = (1/m)*acc_grad1 + (lambda/m)*temp1;
	Theta2_grad = (1/m)*acc_grad2 + (lambda/m)*temp2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
