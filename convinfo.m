function [statusall,gradtest,mtest,ftest]=...
         convinfo(fwdprob,epsr,Ghat1,rhat0,rhat1,m0,m1,verbose,fid)
% CONVINFO - Display convergence info for previous linearization step
% in locally linearized inversion of a nonlinear inverse problem (using the
% "invGN.m" script).
%
% by Andrew Ganse, Applied Physics Laboratory, University of Washington, Seattle.  
% Contact at andy@ganse.org , http://research.ganse.org
% Copyright (c) 2015, University of Washington, via 3-clause BSD license.  
% See LICENSE.txt for full license statement.  
%
% Based on material in:
% RC Aster, B Borchers, and CH Thurber, "Parameter Estimation and Inverse
% Problems," Elsevier Academic Press, 2004.
%
% Usage:
%    [statusall,gradtest,mtest,ftest]=...
%        convinfo(fwdprob,epsr,Ghat1,rhat0,rhat1,m0,m1,verbose,fid);
%
% Inputs:
%     fwdprob = Matlab function handle to forward problem function which
%               is of form outvector=myfunction(invector).  (See example 1
%               of "help function_handle" in Matlab for how to implement)
%        epsr = relative precision of forward problem output (predicted data)
%       Ghat1 = normalized Jacobian matrix of partial derivs of fwdprob at m1,
%               ie Ghat1=C^{-1/2}*G(m1), where C is the cov matrix of meas noise
%       rhat0 = column vector of normalized residuals at m0, i.e.
%               C^{-1/2}*(dmeas-dpred0)
%       rhat1 = column vector of normalized residuals at m1, i.e.
%               C^{-1/2}*(dmeas-dpred1)
%          m0 = column vector containing previous model value.
%          m1 = updated model (column) vector from latest invOccam() iteration.
%     verbose = 0: no status/diagnostic output, 1: some output
%         fid = file identifier for verbose output (status, warnings, etc.)
%
% Output "statusall" is a logical AND of the boolean results of 3 conv tests:
%   * gradtest = obj gradient being close enough to zero per A/B/T def p182
%   * mtest = change in norm(m) being close enough to zero per A/B/T def p182
%   * ftest = change in obje value being close enough to zero per A/B/T def p182
% Statusall=1 means all 3 tests passed.  Statusall=0 means at least 1 didn't,
% in which case you probably want to take another linearization step.
%
% by Andrew Ganse, Applied Physics Laboratory, University of Washington, 2007-2008.
% andy@ganse.org  
% Copyright (c) 2015, University of Washington, via 3-clause BSD license.  
% See LICENSE.txt for full license statement.  

if length(epsr)>1
  fprintf(fid,'      convinfo: note skipping this for case with length(epsr)>1...\n');
  gradtest=0; mtest=0; ftest=0; statusall=0;
  return;
end


f1=rhat1'*rhat1;  % objective function value at m1
f0=rhat0'*rhat0;  % objective function value at m0


% check if the gradient of f(m) is approximately 0:
gradobj1=2*Ghat1'*rhat1;  % objective gradient at m1:  grad(f(m1)) = 2G(m1)^T F(m1)
                    % where G(m1) is the normalized jacobian matrix of fwd problem at m1
left1=norm(gradobj1);
right1=sqrt(epsr)*(1+f1);
gradtest=left1<right1;
if verbose>1
if gradtest
   ans='small enough';
   op='<'; 
else
   ans='not small enough yet';
   op='not <';
end;
fprintf(fid,'      grad(Fobj): %s (%g %s %g)\n', ans, left1, op, right1);
end

% check if successive values of m are close:
left2=norm(m1-m0);
right2=sqrt(epsr)*(1+norm(m1));
mtest=left2<right2;
if verbose>1
if mtest
   ans='close enough';
   op='<'; 
else
   ans='not close enough yet';
   op='not <';
end;
fprintf(fid,'      dm: %s (%g %s %g)\n', ans, left2, op, right2);
end

% check if the values of f(m) have stopped changing:
left3=abs(f1-f0);
right3=epsr*(1+f1);
ftest=left3<right3;
if verbose>1
if ftest
   ans='close enough';
   op='<'; 
else
   ans='not close enough yet';
   op='not <';
end;
fprintf(fid,'      df: %s (%g %s %g)\n', ans, left3, op, right3);
end


statusall = gradtest && mtest && ftest;
%if verbose>=1
%if statusall
%   fprintf(fid,'      CONVERGED according to grad(Fobj) and dm and df...\n');
%else
%   fprintf(fid,'      Not converged yet according to grad(Fobj) and dm and df...\n');
%end
%end
