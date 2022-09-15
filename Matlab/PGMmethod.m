function [B, c,G,determinant] = PGMmethod(A,num_chans,R,eta,epsilon,convergence,maxiter,vargin)
n = size(A,1); % rows
flag=0;
s = warning('error', 'MATLAB:nearlySingularMatrix');
if nargin >= 8

    arg = vargin;

    if strcmp(arg,'norm')
        flag = 1;
        fprintf('gradients normalized to unit length \n')
    else
        flag = 0;
        fprintf('no noramlizaion was applied \n')
    end
end


% initialize B
zero = zeros(size(A));
expA = expm(A');

[vecs,vals] = eig(A');
B = vecs(1:num_chans,:)';

% B = randn(n,num_chans);
B = sqrt(R+epsilon)*B/sqrt(trace(B'*B)); % normalization
broke=0;
f = waitbar(0, 'Convergence');
for i = 1:maxiter
   % compute Grammian
    M = [-A B*B';zeros(size(A)) A'];
    expM = expm(M);
    G = expA'*expM(1:size(A,1),size(A,1)+1:end);
    try
        Ginv = inv(G)';
    catch
       fprintf('Can''t compute %s (reason: %s) \n', 'Ginv', lasterr);
       disp('you cant make an uncontrollable system more controllable! \n')
       broke=1;
       break
    end

    % compute gradients
    dNdB = 4*(trace(B'*B)-R)*B;
    determinant(i) = det(G);
    M = [-A' Ginv;zero A];
    expM = expm(M);
    L = expA'*expM(1:size(A,1),size(A,1)+1:end);
    dEdB = -L*B;
    
    % vectorize gradients
    vdE = dEdB(:);
    vdN = dNdB(:);
    if flag == 1
        vdE = vdE/norm(vdE); % these get way too big - normalize to keep stable
        vdN = vdN/norm(vdN); % these get way too big - normalize to keep stable
    elseif flag ==0
        % no normalization was selected
    end
    % get projection operator
    tmp = eye(numel(vdN))-vdN*pinv(vdN);
    tmp = eta*tmp*vdE;%/norm(vdE); % projections of vdE onto vdN
    
    % update B
    tmp = B - reshape(tmp,size(B));
    B = sqrt(R+epsilon)*tmp/sqrt(trace(tmp'*tmp)); % normalization
    
    % compute the angle between vdE and vdN
    c(i) = 1+((vdE' * vdN) / (sqrt(vdE'*vdE) * sqrt(vdN'*vdN)));
    
    % provide feedback for the user
    status = 1-(c(i) - convergence)/(c(1)-convergence);
    waitbar(status,f)
    
    %end
    if c(i) < convergence
        break
    end
  
end
close(f)
if broke == 0
    fprintf('algorithm finished with a determinant of %g \n' , determinant(end))
elseif broke == 1
    fprintf('an error occured and no results were inidcated \n')
    error('no results were inidcated')
    return
end
warning(s);

