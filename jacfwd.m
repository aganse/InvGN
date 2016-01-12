function [J,F] = jacfwd(funcname,x,epsr,verbose,fid,varargin);
% JACFWD - 1st forward difference approximation of Jacobian matrix
% of partial derivs of function 'funcname' w.r.t. its arguments in vector 'x'.
%             
% by Andrew Ganse, Applied Physics Laboratory, University of Washington, 2005.
% Contact at andy@ganse.org , http://research.ganse.org
% Copyright (c) 2015, University of Washington, via 3-clause BSD license.  
% See LICENSE.txt for full license statement.  
%
% Usage:
% [J,F] = jacfwd('funcname',x,epsf,verbose,fid,...);   
%         [where funcname.m is a file within path]
% or
% [J,F] = jacfwd(@funcname,x,epsf,verbose,fid,...);    
%         [when funcname() is defined within the calling script]
%
% Inputs:
%  funcname = String or function handle for the function being differenced.
%             (See below for required form of this function.)
%         x = The column vector (length M) of arguments to the function funcname.
%             This function assumes that elements of x are scaled to order unity.
%      epsr = The relative accuracy of the forward problem computation - on
%             really simple fwd problem functions, epsr will just equal machine
%             precision.  But for more complicated functions epsr will be larger
%             than machine precision, and determining it may take some analysis.
%             Try e.g. Gill, Murray, & Wright, "Practical Optimization".  In any
%             case note that epsr is used to choose stepsize in the finite diff.
%             Can be scalar (ie same for all elements of model vector) or vector
%             of same length as model vector for 1-to-1 correspondence to difft
%             variable "types" (eg layer velocities vs layer thickness)
%             If EPSR is negative then that flags it as actually being the DX to
%             use.  Note that use of EPSR rather than DX assumes that the values
%             in X have been all scaled to order 1.
%   verbose = 1: output progress text, 0: run quietly.
%             Progress text (percent done) is useful for very slow functions.
%       fid = File handle to which progress text is outputted in verbose option.
%             Note fid=1 outputs to stdout, easiest to do this is not verbose
%             since nothing will then be outputted.
%
% Outputs:
%         J = the finite diffs approximation of NxM Jacobian matrix at x.
%         F = the function output column vector (length N) funcname(x) at x.
%
% Function 'funcname' must be of form:
%   F=funcname(x)
%       or
%   F=funcname(x,...)
%             where the '...' are other arguments required besides the variables
%             x with repsect to which the function is being differenced.


checksize=size(x);
if checksize(2)~=1
   error(' jacfwd(): x vector argument must be a column vector...\n');
%   fprintf(fid,'error: jacfwd(): x vector argument must be a column vector...\n');
%   return;
end
M=length(x);
% The following choice of dx requires that x is scaled to order 1:
% FIXME: should check for that
if epsr(1)>0
  if length(epsr)==1
    dx=epsr^(1/2)*ones(M,1);  % note epsr^(1/2) is for fwd diffs, cent diffs is different value
  elseif length(epsr)==length(x)
    dx=epsr.^(1/2);  % note epsr^(1/2) is for fwd diffs, cent diffs is different value
  end
elseif epsr(1)<0
  if length(epsr)==1
    dx=-epsr*ones(M,1);  % note epsr^(1/2) is for fwd diffs, cent diffs is different value
  elseif length(epsr)==length(x)
    dx=-epsr;  % note epsr^(1/2) is for fwd diffs, cent diffs is different value
  end
else
  disp('      jacfwd: error: all elements of epsr and dx must be nonzero.\n');
  return;
end
% Now make dx and exactly representable number vis a vis machine precision
% so that all error in derivs are from numerator (see Num Rec 2nd ed, sec 5.7)
temp = x+dx;  % note temp is vector of same length as model vector
dx = temp-x;  % Believe it or not this is actually a slightly different dx now,
              % different by an error of order of machine precision.
              % This effect may or may not make it into fwdprob input which 
              % could have limited input precision, but in any case will have
              % an effect via denominator at end of this script.
mask=eye(M);

F=feval(funcname,x,varargin{:});

% forward diffs
for j=1:M
   J(:,j) = ( feval( funcname, x+dx.*mask(:,j), varargin{:} ) - F ) ./ dx(j);

   % percent-done markers to show at matlab cmdline while running:
   if verbose>1
       fprintf(fid,'    ');
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
