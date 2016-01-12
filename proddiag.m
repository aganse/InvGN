function Cdiag = proddiag(A,B)
% PRODDIAG - Compute only diagonal elements of matrix product solution
% Ie, in cases or large outer dimensions of matrices A & B, it may be far too
% slow (or not even feasible) to multiply them completely to obtain the product
% matrix.  If we just want the diagonal elements of their product, we can
% compute that more directly, returning a vector with the diagonal.
%
% by Andrew Ganse, Applied Physics Laboratory, University of Washington, 2010.
% Contact at andy@ganse.org , http://research.ganse.org
% Copyright (c) 2015, University of Washington, via 3-clause BSD license.  
% See LICENSE.txt for full license statement.  
%
% Usage:
% Cdiag = proddiag(A,B)

[Ma,Na]=size(A);
[Mb,Nb]=size(B);
if Na~=Mb
  disp('proddiag: error: must have equal inner dimensions of matrices for mult');
  return;
end

M=min(Ma,Nb);  % ie the diag of the product matrix will only be as long as the
               % smaller of the two outer dimensions of A & B
Cdiag=zeros(M,1);

for i=1:M
  Cdiag(i)=A(i,:)*B(:,i);
end

