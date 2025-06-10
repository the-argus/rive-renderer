#pragma once

#include "draw_path_common.exports.h"

namespace rive {
namespace pls {
namespace glsl {
const char draw_path_common[] = R"===(#ifdef V
O1 Q2(x9,AC);P1 V1 R2(x7,y9,SB);M3(H6,Y8,PB);N3(I6,Z8,HB);R2(y7,z9,FC);W1
#ifdef IC
p m0 L4(int C7){return m0(C7&((1<<k7)-1),C7>>k7);}p float D7(A a0,d A9){d R0=h0(a0,A9);return(abs(R0.x)+abs(R0.y))*(1./dot(R0,R0));}p bool F6(g a4,g N5,int L,x2(M)S2,x2(d)B9
#ifndef CB
,x2(l0)T2
#else
,x2(M)C9
#endif
U2){int M4=int(a4.x);float S1=a4.y;float N4=a4.z;int E7=floatBitsToInt(a4.w)>>2;int O5=floatBitsToInt(a4.w)&3;int P5=min(M4,E7-1);int O4=L*E7+P5;T z3=I1(AC,L4(O4));uint D=z3.w;T Q5=z0(FC,a7(D));d F7=uintBitsToFloat(Q5.xy);S2=O0(Q5.z&0xffffu);uint D9=Q5.w;A a0=W0(uintBitsToFloat(z0(SB,S2*2u)));T c4=z0(SB,S2*2u+1u);d K0=uintBitsToFloat(c4.xy);float J1=uintBitsToFloat(c4.z);
#ifdef CB
C9=O0(c4.w);
#endif
uint G7=D&I4;if(G7!=0u){M4=int(N5.x);S1=N5.y;N4=N5.z;}if(M4!=P5){O4+=M4-P5;T H7=I1(AC,L4(O4));if((H7.w&0xffffu)!=(D&0xffffu)){bool E9=J1==.0||F7.x!=.0;if(E9){z3=I1(AC,L4(int(D9)));}}else{z3=H7;}D=z3.w|G7;}float p1=uintBitsToFloat(z3.z);d V2=d(sin(p1),-cos(p1));d I7=uintBitsToFloat(z3.xy);d R5;if(J1!=.0){S1*=sign(determinant(a0));if((D&J4)!=0u)S1=min(S1,.0);if((D&p7)!=0u)S1=max(S1,.0);float y2=D7(a0,V2)*k2;h J7=1.;if(y2>J1){J7=E0(J1)/E0(y2);J1=y2;}d W2=h0(V2,J1+y2);
#ifndef CB
float x=S1*(J1+y2);T2=Z1((1./(y2*2.))*(d(x,-x)+J1)+.5);
#endif
uint S5=D&Z3;if(S5!=0u){int d4=2;if((D&M5)==0u)d4=-d4;if((D&I4)!=0u)d4=-d4;m0 F9=L4(O4+d4);T G9=I1(AC,F9);float H9=uintBitsToFloat(G9.z);float e4=abs(H9-p1);if(e4>E4)e4=2.*E4-e4;bool P4=(D&M5)!=0u;bool I9=(D&J4)!=0u;float K7=e4*(P4==I9?-.5:.5)+p1;d Q4=d(sin(K7),-cos(K7));float T5=D7(a0,Q4);float f4=cos(e4*.5);float U5;if((S5==v9)||(S5==w9&&f4>=.25)){float J9=(D&H4)!=0u?1.:.25;U5=J1*(1./max(f4,J9));}else{U5=J1*f4+T5*.5;}float V5=U5+T5*k2;if((D&o7)!=0u){float L7=J1+y2;float K9=y2*.125;if(L7<=V5*f4+K9){float L9=L7*(1./f4);W2=Q4*L9;}else{d W5=Q4*V5;d M9=d(dot(W2,W2),dot(W5,W5));W2=h0(M9,inverse(A(W2,W5)));}}d N9=abs(S1)*W2;float M7=(V5-dot(N9,Q4))/(T5*(k2*2.));
#ifndef CB
if((D&J4)!=0u)T2.y=E0(M7);else T2.x=E0(M7);
#endif
}
#ifndef CB
T2*=J7;T2.y=max(T2.y,E0(1e-4));
#endif
R5=h0(a0,S1*W2);if(O5!=q7)return false;}else{if(O5==v7)I7=F7;R5=sign(h0(S1*V2,inverse(a0)))*k2;if((D&I4)!=0u)N4=-N4;
#ifndef CB
T2=Z1(N4,-1);
#endif
if((D&n7)!=0u&&O5!=r7)return false;}B9=h0(a0,I7)+R5+K0;return true;}
#endif
#ifdef DB
p d G6(D1 X5,x2(M)S2,x2(h)O9 U2){S2=O0(floatBitsToUint(X5.z)&0xffffu);A a0=W0(uintBitsToFloat(z0(SB,S2*2u)));T c4=z0(SB,S2*2u+1u);d K0=uintBitsToFloat(c4.xy);O9=float(floatBitsToInt(X5.z)>>16)*sign(determinant(a0));return h0(a0,X5.xy)+K0;}
#endif
#endif
)===";
} // namespace glsl
} // namespace pls
} // namespace rive