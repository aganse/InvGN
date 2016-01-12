function [J,F] = jaccen(funcname,x,epsf,verbose,fid,varargin);
% JACCEN - 1st centered difference approximation of Jacobian matrix
% of partial derivs of function 'funcname' w.r.t. its arguments in vector 'x'.
%             
% by Andrew Ganse, Applied Physics Laboratory, University of Washington, 2005.
% Contact at andy@ganse.org , http://research.ganse.org
% Copyright (c) 2015, University of Washington, via 3-clause BSD license.  
% See LICENSE.txt for full license statement.  
%
% Usage:
% [J,F] = jaccen('funcname',x,dx,verbose,fid,...);   
%         [where funcname.m is a file within path]
% or
% [J,F] = jaccen(@funcname,x,dx,verbose,fid,...);    
%         [when funcname() is defined within the calling script]
%
% Inputs:
%  funcname = String or function handle for the function being differenced.
%             (See below for required form of this function.)
%         x = The column vector (length M) of arguments to the function funcname.
%        dx = Small change in x used to compute differences.
%             dx can be a scalar or vector of same length as x.
%             If scalar it is converted to a vector of same length of x in
%             which all elements are equal to the scalar dx.
%             For problems in which x is all the same type of quantity, such
%             as a profile of wavespeeds, it is recommended to use a scalar
%             dx = sqrt(epsilon), where epsilon is the accuracy of funcname.
%   verbose = 1: output progress text, 0: run quietly.
%             Progress text (percent done) is useful for very slow functions.
%       fid = File handle to which progress text is outputted in verbose option.
%             Note fid=1 outputs to stdout, easiest to do this is not verbose
%             since nothing will then be outputted.
%
% Outputs:
%         J = the finite diffs approximation of NxM Jacobian matrix at x
%         F = the function output column vector (length N) funcname(x) at x
%
% Function 'funcname' must be of form:
%   F=funcname(x)
%       or
%   F=funcname(x,...)
%             where the '...' are other arguments required besides the variables
%             x with repsect to which the function is being differenced.


checksize=size(x);
if checksize(2)~=1
   error(' jac(): x vector argument must be a column vector...\n');
%   fprintf(fid,'error: jac(): x vector argument must be a column vector...\n');
%   return;
end
M=length(x);

% The following choice of dx requires that x is scaled to order 1:
% FIXME: should check for that
dx=epsf^(1/3)*ones(M,1);  % note epsf^(1/3) is for ctr diffs
% Now make dx and exactly representable number vis a vis machine precision
% so that all error in derivs are from numerator (see Num Rec 2nd ed, sec 5.7)
temp = x+dx;
dx = temp-x;  % Believe it or not this is actually a slightly different dx now,
              % different by an error of order of machine precision.
              % This effect may or may not make it into fwdprob input which 
              % could have limited input precision, but in any case will have
              % an effect via denominator at end of this script.
mask=eye(M);

if verbose>1
   fprintf(fid,'  Computing 1st centered finite diffs for Jacobian matrix...\n');
end

if nargout==2
   F=feval(funcname,x,varargin{:});
end

% centered diffs
for j=1:M
   F1=feval(funcname,x+dx.*mask(:,j),varargin{:});
   F2=feval(funcname,x-dx.*mask(:,j),varargin{:});
   J(:,j) = ( F1 - F2 ) ./ dx(j)/2;

   % percent-done markers to show at matlab cmdline while running:
   if verbose>0
       p=j/M;
       if round(mod(p*100,10))==0
           fprintf(fid,'  ');
           fprintf(fid,'%2.0f%%..',p*100);
           if p==1
               fprintf(fid,'\n');
           end;
       end;
   end;

end

