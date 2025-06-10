#pragma once

#include "tessellate.exports.h"

namespace rive {
namespace pls {
namespace glsl {
const char tessellate[] = R"===(#define va 10
#ifdef V
U0(P)q0(0,g,GC);q0(1,g,HC);q0(2,g,ZB);q0(3,T,LB);V0
#endif
A1 k0 I(0,g,C3);k0 I(1,g,D3);k0 I(2,g,A2);k0 I(3,D1,e3);e6 I(4,uint,l4);B1 p A j8(d T0,d r0,d I0,d f1){A t;t[0]=(any(notEqual(T0,r0))?r0:any(notEqual(r0,I0))?I0:f1)-T0;t[1]=f1-(any(notEqual(f1,I0))?I0:any(notEqual(I0,r0))?r0:T0);return t;}
#ifdef V
O1 P1 V1 R2(x7,y9,SB);R2(y7,z9,FC);W1 float k8(d m,d b){float wa=dot(m,b);float l8=dot(m,m)*dot(b,b);return(l8==.0)?1.:clamp(wa*inversesqrt(l8),-1.,1.);}g1(UD,P,r,j,L){v0(L,r,GC,g);v0(L,r,HC,g);v0(L,r,ZB,g);v0(L,r,LB,T);Q(C3,g);Q(D3,g);Q(A2,g);Q(e3,D1);Q(l4,uint);d T0=GC.xy;d r0=GC.zw;d I0=HC.xy;d f1=HC.zw;bool m8=j<4;float y=m8?ZB.z:ZB.w;int n6=int(m8?LB.x:LB.y);
#ifdef T9
int n8=n6<<16;if(LB.z==0xffffffffu){--n8;}float X4=float(n8>>16);
#else
float X4=float(n6<<16>>16);
#endif
float Y4=float(n6>>16);d j1=d((j&1)==0?X4:Y4,(j&2)==0?y+1.:y);uint v1=LB.z&0x3ffu;uint o8=(LB.z>>10)&0x3ffu;uint f3=LB.z>>20;uint D=LB.w;if(Y4<X4){D|=I4;}if((Y4-X4)*v.G5<.0){j1.y=2.*y+1.-j1.y;}if((D&r9)!=0u){uint xa=z0(FC,a7(D)).z;A p8=W0(uintBitsToFloat(z0(SB,xa*2u)));d q8=h0(p8,-2.*r0+I0+T0);d r8=h0(p8,-2.*I0+f1+r0);float e1=max(dot(q8,q8),dot(r8,r8));float E2=max(ceil(sqrt(.75*4.*sqrt(e1))),1.);v1=min(uint(E2),v1);}uint Z4=v1+o8+f3-1u;A U1=j8(T0,r0,I0,f1);float p1=acos(k8(U1[0],U1[1]));float g2=p1/float(o8);float o6=determinant(A(I0-T0,f1-r0));if(o6==.0)o6=determinant(U1);if(o6<.0)g2=-g2;C3=g(T0,r0);D3=g(I0,f1);A2=g(float(Z4)-abs(Y4-j1.x),float(Z4),(f3<<10)|v1,g2);if(f3>1u){A p6=A(U1[1],ZB.xy);float ya=acos(k8(p6[0],p6[1]));float v8=float(f3);if((D&(Z3|H4))==H4){v8-=2.;}float q6=ya/v8;if(determinant(p6)<.0)q6=-q6;e3.xy=ZB.xy;e3.z=q6;}l4=D;g B;B.x=j1.x*(2./q9)-1.;B.y=j1.y*v.G5-sign(v.G5);B.zw=d(0,1);S(C3);S(D3);S(A2);S(e3);S(l4);h1(B);}
#endif
#ifdef GB
r2(T,VD){N(C3,g);N(D3,g);N(A2,g);N(e3,D1);N(l4,uint);d T0=C3.xy;d r0=C3.zw;d I0=D3.xy;d f1=D3.zw;A U1=j8(T0,r0,I0,f1);float za=max(floor(A2.x),.0);float Z4=A2.y;uint w8=uint(A2.z);float v1=float(w8&0x3ffu);float f3=float(w8>>10);float g2=A2.w;uint D=l4;float g3=Z4-f3;float w1=za;if(w1<=g3){D&=~Z3;}else{T0=r0=I0=f1;U1=A(U1[1],e3.xy);v1=1.;w1-=g3;g3=f3;if((D&Z3)!=0u){if(w1<2.5)D|=M5;if(w1>1.5&&w1<3.5)D|=o7;}else if((D&H4)!=0u){g3-=2.;w1--;}g2=e3.z;D|=g2<.0?J4:p7;}d a5;float p1=.0;if(w1==.0||w1==g3||(D&Z3)!=0u){bool P4=w1<g3*.5;a5=P4?T0:f1;p1=d7(P4?U1[0]:U1[1]);}else if((D&n7)!=0u){a5=r0;}else{float T1,h3;if(v1==g3){T1=w1/v1;h3=.0;}else{d d0,g0,T4=r0-T0;d c8=f1-T0;d x8=I0-r0;g0=x8-T4;d0=-3.*x8+c8;d Aa=g0*(v1*2.);d Ba=T4*(v1*v1);float c5=.0;float Ca=min(v1-1.,w1);d r6=normalize(U1[0]);float Da=-abs(g2);float Ea=(1.+w1)*abs(g2);for(int G3=va-1;G3>=0;--G3){float m4=c5+exp2(float(G3));if(m4<=Ca){d v6=m4*d0+Aa;v6=m4*v6+Ba;float Fa=dot(normalize(v6),r6);float w6=m4*Da+Ea;w6=min(w6,E4);if(Fa>=cos(w6))c5=m4;}}float Ga=c5/v1;float y8=w1-c5;float d5=acos(clamp(r6.x,-1.,1.));d5=r6.y>=.0?d5:-d5;p1=y8*g2+d5;d V2=d(sin(p1),-cos(p1));float m=dot(V2,d0),e5=dot(V2,g0),e0=dot(V2,T4);float Ha=max(e5*e5-m*e0,.0);float B2=sqrt(Ha);if(e5>.0)B2=-B2;B2-=e5;float z8=-.5*B2*m;d x6=(abs(B2*B2+z8)<abs(m*e0+z8))?d(B2,m):d(e0,B2);h3=(x6.y!=.0)?x6.x/x6.y:.0;h3=clamp(h3,.0,1.);if(y8==.0)h3=.0;T1=max(Ga,h3);}d Ia=r3(T0,r0,T1);d A8=r3(r0,I0,T1);d Ja=r3(I0,f1,T1);d B8=r3(Ia,A8,T1);d C8=r3(A8,Ja,T1);a5=r3(B8,C8,T1);if(T1!=h3)p1=d7(C8-B8);}v2(T(floatBitsToUint(D1(a5,p1)),D));}
#endif
)===";
} // namespace glsl
} // namespace pls
} // namespace rive