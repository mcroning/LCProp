ClearAll["Global`*"];
out=Last[Rest[$ScriptCommandLine]];
result=TimeConstrained[
 n=4;nn=16;
 dd=Table[If[i==j,0,(-1)^(i-j)/2 Cot[Pi(i-j)/n]],{i,0,n-1},{j,0,n-1}];
 dx=KroneckerProduct[dd,IdentityMatrix[n]];dy=KroneckerProduct[IdentityMatrix[n],dd];ell=-dx.dx-dy.dy;
 g=IdentityMatrix[nn]-ConstantArray[1/nn,{nn,nn}];
 target=Flatten[Table[Cos[2Pi i/n]/100+Sin[2Pi j/n]/200+(-1)^i/50+(-1)^(i+j)/100,{i,0,n-1},{j,0,n-1}]];
 carrier=ConstantArray[1,nn]+ell.target;ii=Exp[-target]/carrier;
 zz=Array[z,4];weights=Array[b,4];pp=4 weights Exp[-zz]/Total[weights Exp[-zz]];
 (* Derive normalized-carrier Jacobian independently, with symbolic positive weights. *)
 jSymbol=Table[D[-pp[[i]],zz[[j]]],{i,4},{j,4}];
 jExpected=DiagonalMatrix[pp]-Outer[Times,pp,pp]/4;
 symbolic=And@@Flatten[Map[TrueQ[Simplify[#]==0]&,jSymbol-jExpected,{2}]];
 rr[v_]:=Module[{p=Exp[-v]/N[ii,100]},p=nn p/Total[p];ConstantArray[1,nn]+ell.v-p];
 hh[v_]:=Module[{p=Exp[-v]/N[ii,100]},p=nn p/Total[p];ell+DiagonalMatrix[p]-Outer[Times,p,p]/nn];
 v=ConstantArray[SetPrecision[0,100],nn];steps=0;
 While[Max[Abs[rr[v]]]>10^-80 && steps<12,delta=LinearSolve[hh[v]+ConstantArray[1/nn,{nn,nn}],-rr[v]];v=g.(v+delta);steps++];
 If[Max[Abs[rr[v]]]>10^-80,Exit[5]];
 <|"symbolicJacobian"->symbolic,"meanZeroRank"->MatrixRank[ell+IdentityMatrix[nn]-ConstantArray[1/nn,{nn,nn}]],"shape"->{4,4},"dx"->N[Pi/2],"dy"->N[Pi/2],"intensity"->N[ii,17],"target"->N[target,17],"solution"->N[v,17],"iterations"->steps,"residualMax100digit"->ToString[Max[Abs[rr[v]]],InputForm],"solutionError100digit"->ToString[Max[Abs[v-N[target,100]]],InputForm],"positiveCarrier"->TrueQ[Min[carrier]>0]|>,90,$Failed];
If[result===$Failed || !TrueQ[result["symbolicJacobian"]],Exit[2]];
If[Export[out,result,"RawJSON"]===$Failed,Exit[3]];Print["PASS independent constant-gauge reference"];
