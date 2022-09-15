%% find optimal input-gain matrix to minimize control costs
% Given: network adjacency matrix, radius eta, epsilon
% convergence, maxiter

% define state-space model
% x_dot = A*x + B*u
% y = C*x+D*u

close all

numstates = 3; % number of hidden states
numinputs = 2; % number of inputs
numchans = 60; % number of EEG electrodes
srate = 1e3; % sample frequency of the simulation

% generate random state-space model
rng(0)
A = rand(numstates); % state gain
A = [0 1 1;1 0 0;0 1 0];
A = round((A + A')/2);
A = A - diag(diag(A));

D = diag(1./sqrt(sum(A)));
A = D*(A+eye(numstates))*D;

% not necessary for the current project
C = 10*randn(numchans,numstates); % observation
D = 10*randn(numchans,numinputs); % feed through

eta = .001; % step size
epsilon = .01; % deviation from the sphere
convergence = 0.00001; % convergence criteria

maxiter = 60000; % max number of allowed iterations
R =10*sqrt(trace(A'*A)); % radius of the sphere
%RA =trace(A'*A)+.001; % radius of the sphere
[B,c,G,determinant]=PGMmethod(A,numinputs,R,eta,epsilon,convergence,maxiter,'norm');
