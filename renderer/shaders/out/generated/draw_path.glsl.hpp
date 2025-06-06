#pragma once

#include "draw_path.exports.h"

namespace rive {
namespace pls {
namespace glsl {
const char draw_path[] = R"===(#ifdef V
U0(P)
#ifdef DB
q0(0,H3,KB);
#else
q0(0,g,UB);q0(1,g,VB);
#endif
V0
#endif
A1 k0 I(0,g,o0);
#ifndef CB
#ifdef DB
FB I(1,h,C1);
#else
k0 I(2,l0,J0);
#endif
FB I(3,h,W);
#ifdef BB
FB I(4,h,G0);
#endif
#ifdef Z
k0 I(5,g,i0);
#endif
#endif
#ifdef AB
FB I(6,h,z2);
#endif
B1
#ifdef V
g1(OB,P,r,j,L){
#ifdef DB
v0(j,r,KB,D1);
#else
v0(j,r,UB,g);v0(j,r,VB,g);
#endif
Q(o0,g);
#ifndef Wa
#ifdef DB
Q(C1,h);
#else
Q(J0,l0);
#endif
Q(W,h);
#ifdef BB
Q(G0,h);
#endif
#ifdef Z
Q(i0,g);
#endif
#endif
#ifdef AB
Q(z2,h);
#endif
bool N7=false;M k1;d R;
#ifdef CB
M O7;
#endif
#ifdef DB
R=G6(KB,k1,C1 j3);
#else
N7=!F6(UB,VB,L,k1,R
#ifndef CB
,J0
#else
,O7
#endif
j3);
#endif
x0 E=q2(PB,k1);
#ifndef CB
W=F4(k1,v.v3);if((E.x&M6)!=0u)W=-W;
#endif
uint H1=E.x&0xfu;
#ifdef BB
uint P9=(H1==y4?E.y:E.x)>>16;G0=F4(P9,v.v3);if(H1==y4)G0=-G0;
#endif
#ifdef AB
z2=float((E.x>>4)&0xfu);
#endif
d g4=R;
#ifdef Q9
g4.y=float(v.o9)-g4.y;
#endif
#ifdef Z
A X0=W0(z0(HB,k1*4u+2u));g i1=z0(HB,k1*4u+3u);
#ifndef CB
i0=n4(X0,i1.xy,g4);
#else
j7(X0,i1.xy,g4);
#endif
#endif
if(H1==N6){i f=unpackUnorm4x8(E.y);o0=g(f);}
#ifdef BB
else if(H1==y4){h R4=F4(E.x>>16,v.v3);o0=g(R4,0,0,0);}
#endif
else{A R9=W0(z0(HB,k1*4u));g Y5=z0(HB,k1*4u+1u);d o2=h0(R9,g4)+Y5.xy;if(H1==x4||H1==O6){o0.w=-uintBitsToFloat(E.y);if(Y5.z>.9){o0.z=2.;}else{o0.z=Y5.w;}if(H1==x4){o0.y=.0;o0.x=o2.x;}else{o0.z=-o0.z;o0.xy=o2.xy;}}else{float J2=uintBitsToFloat(E.y);o0=g(o2.x,o2.y,J2,-2.);}}g B;if(!N7){B=i2(R);
#ifdef CB
B.z=K5(O7);
#endif
}else{B=g(v.j2,v.j2,v.j2,v.j2);}S(o0);
#ifndef CB
#ifdef DB
S(C1);
#else
S(J0);
#endif
S(W);
#ifdef BB
S(G0);
#endif
#ifdef Z
S(i0);
#endif
#endif
#ifdef AB
S(z2);
#endif
h1(B);}
#endif
#ifdef GB
F2 x1(q4,DC);x1(k3,NB);
#ifdef CB
#ifdef AB
x1(A7,EC);
#endif
#endif
G2 J3(q4,v5)l3(k3,n2)L3 O3 p i P7(g K1
#ifdef EB
,d Z5,d a6
#endif
w4){if(K1.w>=.0){return M0(K1);}else if(K1.w>-1.){float t=K1.z>.0?K1.x:length(K1.xy);t=clamp(t,.0,1.);float Q7=abs(K1.z);float x=Q7>1.?(1.-1./L5)*t+(.5/L5):(1./L5)*t+Q7;float S9=-K1.w;return M0(R3(DC,v5,d(x,S9),.0));}else{i f;
#ifdef EB
f=Q3(NB,n2,K1.xy,Z5,a6);
#else
f=M2(NB,n2,K1.xy);
#endif
f.w*=K1.z;return f;}}
#ifndef CB
E1 C0(w5,c0);D0(x5,Q0);
#ifdef BB
D0(y5,w0);
#endif
C0(w7,w2);F1 R1(IB){N(o0,g);
#ifdef DB
N(C1,h);
#else
N(J0,l0);
#endif
N(W,h);
#ifdef BB
N(G0,h);
#endif
#ifdef Z
N(i0,g);
#endif
#ifdef AB
N(z2,h);
#endif
#ifdef EB
d Z5=dFdx(o0.xy);d a6=dFdy(o0.xy);
#endif
#ifndef DB
n1;
#endif
l0 R7=unpackHalf2x16(L0(Q0));h S7=R7.y;h Y0=S7==W?R7.x:E0(0);
#ifdef DB
Y0+=C1;
#else
if(J0.y>=.0)Y0=max(min(J0.x,J0.y),Y0);else Y0+=J0.x;a1(Q0,packHalf2x16(Z1(Y0,W)));
#endif
h C=abs(Y0);
#ifdef OC
if(W<.0)C=1.-abs(fract(C*.5)*2.+-1.);
#endif
C=min(C,E0(1));
#ifdef BB
if(G0<.0){h G1=-G0;
#ifdef GD
h R4=o0.x;if(R4!=.0){l0 Z0=unpackHalf2x16(L0(w0));h y3=Z0.y;h S4;if(y3!=G1){S4=y3==R4?Z0.x:.0;
#ifndef DB
F0(w2,M0(S4,0,0,0));
#endif
}else{S4=N0(w2).x;
#ifndef DB
y0(w2);
#endif
}C=min(C,S4);}
#endif
a1(w0,packHalf2x16(Z1(C,G1)));y0(c0);}else
#endif
{
#ifdef BB
if(G0!=.0){l0 Z0=unpackHalf2x16(L0(w0));h y3=Z0.y;h I2=y3==G0?Z0.x:E0(0);C=min(C,I2);}y0(w0);
#endif
#ifdef Z
h K2=C5(M0(i0));C=clamp(K2,E0(0),C);
#endif
i f=P7(o0
#ifdef EB
,Z5,a6
#endif
L2);f.w*=C;i y1;if(S7!=W){y1=N0(c0);
#ifndef DB
F0(w2,y1);
#endif
}else{y1=N0(w2);
#ifndef DB
y0(w2);
#endif
}
#ifdef AB
if(z2!=E0(U6)){
#ifdef JB
f=E3(
#else
f=F3(
#endif
f,S3(y1),O0(z2));}else
#endif
{
#ifndef TC
f.xyz*=f.w;f=f+y1*(1.-f.w);
#endif
}F0(c0,f);}
#ifndef DB
o1;
#endif
c2;}
#else
r2(i,IB){N(o0,g);
#ifdef AB
N(z2,h);
#endif
i f=P7(o0);
#ifdef AB
i y1=I1(EC,m0(floor(n0.xy)));
#ifdef JB
f=E3(
#else
f=F3(
#endif
f,S3(y1),O0(z2));
#else
f=p2(f);
#endif
v2(f);}
#endif
#endif
)===";
} // namespace glsl
} // namespace pls
} // namespace rive