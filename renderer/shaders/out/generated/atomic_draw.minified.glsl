#ifdef DRAW_PATH
#ifdef VERTEX
U0(P)q0(0,g,UB);q0(1,g,VB);V0
#endif
A1 k0 I(0,l0,J0);OPTIONALLY_FLAT I(1,M,W);B1
#ifdef VERTEX
g1(OB,P,r,j,L){v0(j,r,UB,g);v0(j,r,VB,g);Q(J0,l0);Q(W,M);g B;d R;if(F6(UB,VB,L,W,R,J0 j3)){B=i2(R);}else{B=g(v.j2,v.j2,v.j2,v.j2);}S(J0);S(W);h1(B);}
#endif
#endif
#ifdef DRAW_INTERIOR_TRIANGLES
#ifdef VERTEX
U0(P)q0(0,H3,KB);V0
#endif
A1 OPTIONALLY_FLAT I(0,h,C1);OPTIONALLY_FLAT I(1,M,W);B1
#ifdef VERTEX
g1(OB,P,r,j,L){v0(j,r,KB,D1);Q(C1,h);Q(W,M);d R=G6(KB,W,C1 j3);g B=i2(R);S(C1);S(W);h1(B);}
#endif
#endif
#ifdef DRAW_IMAGE
#ifdef DRAW_IMAGE_RECT
#ifdef VERTEX
U0(P)q0(0,g,MB);V0
#endif
A1 k0 I(0,d,B0);k0 I(1,h,D2);
#ifdef ENABLE_CLIP_RECT
k0 I(2,g,i0);
#endif
B1
#ifdef VERTEX
O1 P1 V1 W1 l5(OB,P,r,j,L){v0(j,r,MB,g);Q(B0,d);Q(D2,h);
#ifdef ENABLE_CLIP_RECT
Q(i0,g);
#endif
bool m5=MB.z==.0||MB.w==.0;D2=m5?.0:1.;d R=MB.xy;A a0=W0(J.n5);A I3=transpose(inverse(a0));if(!m5){float o5=k2*p5(I3[1])/dot(a0[1],I3[1]);if(o5>=.5){R.x=.5;D2*=.5/o5;}else{R.x+=o5*MB.z;}float q5=k2*p5(I3[0])/dot(a0[0],I3[0]);if(q5>=.5){R.y=.5;D2*=.5/q5;}else{R.y+=q5*MB.w;}}B0=R;R=h0(a0,R)+J.K0;if(m5){d E2=h0(I3,MB.zw);E2*=p5(E2)/dot(E2,E2);R+=k2*E2;}
#ifdef ENABLE_CLIP_RECT
i0=n4(W0(J.X0),J.i1,R);
#endif
g B=i2(R);S(B0);S(D2);
#ifdef ENABLE_CLIP_RECT
S(i0);
#endif
h1(B);}
#endif
#else
#ifdef VERTEX
U0(X1)q0(0,d,WB);V0 U0(l2)q0(1,d,XB);V0
#endif
A1 k0 I(0,d,B0);
#ifdef ENABLE_CLIP_RECT
k0 I(1,g,i0);
#endif
B1
#ifdef VERTEX
o4(OB,X1,Y1,l2,m2,j){v0(j,Y1,WB,d);v0(j,m2,XB,d);Q(B0,d);
#ifdef ENABLE_CLIP_RECT
Q(i0,g);
#endif
A a0=W0(J.n5);d R=h0(a0,WB)+J.K0;B0=XB;
#ifdef ENABLE_CLIP_RECT
i0=n4(W0(J.X0),J.i1,R);
#endif
g B=i2(R);S(B0);
#ifdef ENABLE_CLIP_RECT
S(i0);
#endif
h1(B);}
#endif
#endif
#endif
#ifdef DRAW_RENDER_TARGET_UPDATE_BOUNDS
#ifdef VERTEX
U0(P)V0
#endif
A1 B1
#ifdef VERTEX
O1 P1 V1 W1 g1(OB,P,r,j,L){m0 j1;j1.x=(j&1)==0?v.p4.x:v.p4.z;j1.y=(j&2)==0?v.p4.y:v.p4.w;g B=i2(d(j1));h1(B);}
#endif
#endif
#ifdef ENABLE_BINDLESS_TEXTURES
#define r5
#endif
#ifdef DRAW_IMAGE
#define r5
#endif
#ifdef FRAGMENT
F2 x1(q4,DC);
#ifdef r5
x1(k3,NB);
#endif
G2 J3(q4,v5)
#ifdef r5
l3(k3,n2)
#endif
E1
#ifdef ENABLE_ADVANCED_BLEND
#ifdef FRAMEBUFFER_PLANE_IDX_OVERRIDE
C0(NC,c0);
#else
C0(w5,c0);
#endif
#endif
K3(x5,Q0);
#ifdef ENABLE_CLIPPING
D0(y5,w0);
#endif
F1 L3 M3(H6,Y8,PB);N3(I6,Z8,HB);O3 uint J6(float x){return uint(x*K6+P3);}float r4(uint x){return float(x)*L6+(-P3*L6);}i v4(h Y0,x0 E,uint k1 w4 H2){h C=abs(Y0);
#ifdef ENABLE_EVEN_ODD
if((E.x&M6)!=0u)C=1.-abs(fract(C*.5)*2.+-1.);
#endif
C=min(C,E0(1));
#ifdef ENABLE_CLIPPING
uint G1=E.x>>16u;if(G1!=0u){uint Z0=L0(w0);h I2=G1==(Z0>>16u)?unpackHalf2x16(Z0).x:.0;C=min(C,I2);}
#endif
i f=M0(0,0,0,0);uint H1=E.x&0xfu;switch(H1){case N6:f=unpackUnorm4x8(E.y);
#ifdef ENABLE_CLIPPING
y0(w0);
#endif
break;case x4:case O6:
#ifdef ENABLE_BINDLESS_TEXTURES
case P6:
#endif
{A a0=W0(z0(HB,k1*4u));g K0=z0(HB,k1*4u+1u);d o2=h0(a0,n0)+K0.xy;
#ifdef ENABLE_BINDLESS_TEXTURES
if(H1==P6){f=Q3(sampler2D(floatBitsToUint(K0.zw)),n2,o2,a0[0],a0[1]);float J2=uintBitsToFloat(E.y);f.w*=J2;}else
#endif
{float t=H1==x4?o2.x:length(o2);t=clamp(t,.0,1.);float x=t*K0.z+K0.w;float y=uintBitsToFloat(E.y);f=M0(R3(DC,v5,d(x,y),.0));}
#ifdef ENABLE_CLIPPING
y0(w0);
#endif
break;}
#ifdef ENABLE_CLIPPING
case y4:a1(w0,E.y|packHalf2x16(Z1(C,0)));break;
#endif
}
#ifdef ENABLE_CLIP_RECT
if((E.x&a9)!=0u){A a0=W0(z0(HB,k1*4u+2u));g K0=z0(HB,k1*4u+3u);d c9=h0(a0,n0)+K0.xy;l0 Q6=Z1(abs(c9)*K0.zw-K0.zw);h K2=clamp(min(Q6.x,Q6.y)+.5,.0,1.);C=min(C,K2);}
#endif
f.w*=C;return f;}i R6(i S6,i Q1){return S6+Q1*(1.-S6.w);}
#ifdef ENABLE_ADVANCED_BLEND
i z5(i T6,i Q1,M a2){if(a2!=U6){
#ifdef ENABLE_HSL_BLEND_MODES
return E3(
#else
return F3(
#endif
T6,S3(Q1),a2);}else{return R6(p2(T6),Q1);}}i V6(i f,x0 E H2){i Q1=N0(c0);M a2=O0((E.x>>4)&0xfu);return z5(f,Q1,a2);}void A5(i f,x0 E H2){if(f.w!=.0){i d9=V6(f,E l1);F0(c0,d9);}else{y0(c0);}}
#endif
#ifdef ENABLE_ADVANCED_BLEND
#define T3 R1
#define W6 U3
#define m3 c2
#else
#define T3 n3
#define W6 z4
#define m3 V3
#endif
#ifdef DRAW_PATH
T3(IB){N(J0,l0);N(W,M);
#ifndef ENABLE_ADVANCED_BLEND
P0=M0(0,0,0,0);
#endif
h C=min(min(J0.x,abs(J0.y)),E0(1));uint A4=J6(C);uint B5=(X6(W)<<16)|A4;uint m1=B4(Q0,B5);M c1=O0(m1>>16);if(c1!=W){h Y0=r4(m1&0xffffu);x0 E=q2(PB,c1);i f=v4(Y0,E,c1 L2 l1);
#ifdef ENABLE_ADVANCED_BLEND
A5(f,E l1);
#else
P0=p2(f);
#endif
}else{if(J0.y<.0){if(m1<B5){A4+=m1-B5;}A4-=uint(P3);C4(Q0,A4);}discard;}m3}
#endif
#ifdef DRAW_INTERIOR_TRIANGLES
T3(IB){N(C1,h);N(W,M);
#ifndef ENABLE_ADVANCED_BLEND
P0=M0(0,0,0,0);
#endif
h C=C1;uint m1=o3(Q0);M c1=O0(m1>>16);h Y6=r4(m1&0xffffu);if(c1!=W){x0 E=q2(PB,c1);i f=v4(Y6,E,c1 L2 l1);
#ifdef ENABLE_ADVANCED_BLEND
A5(f,E l1);
#else
P0=p2(f);
#endif
}else{C+=Y6;}p3(Q0,(X6(W)<<16)|J6(C));if(c1==W){discard;}m3}
#endif
#ifdef DRAW_IMAGE
W6(IB){N(B0,d);
#ifdef DRAW_IMAGE_RECT
N(D2,h);
#endif
#ifdef ENABLE_CLIP_RECT
N(i0,g);
#endif
i q3=M2(NB,n2,B0);h N2=1.;
#ifdef DRAW_IMAGE_RECT
N2=min(D2,N2);
#endif
#ifdef ENABLE_CLIP_RECT
h K2=C5(M0(i0));N2=clamp(K2,E0(0),N2);
#endif
#ifdef DRAW_IMAGE_MESH
n1;
#endif
uint m1=o3(Q0);h Y0=r4(m1&0xffffu);M c1=O0(m1>>16);x0 Z6=q2(PB,c1);i D5=v4(Y0,Z6,c1 L2 l1);
#ifdef ENABLE_CLIPPING
if(J.G1!=0u){D4(w0);uint Z0=L0(w0);uint G1=Z0>>16;h I2=G1==J.G1?unpackHalf2x16(Z0).x:.0;N2=min(N2,I2);}
#endif
q3.w*=N2*J.J2;
#ifdef ENABLE_ADVANCED_BLEND
if(D5.w!=.0||q3.w!=.0){i Q1=N0(c0);M e9=O0((Z6.x>>4)&0xfu);M f9=O0(J.a2);Q1=z5(D5,Q1,e9);q3=z5(q3,Q1,f9);F0(c0,q3);}else{y0(c0);}
#else
P0=R6(p2(q3),p2(D5));
#endif
p3(Q0,uint(P3));
#ifdef DRAW_IMAGE_MESH
o1;
#endif
m3}
#endif
#ifdef INITIALIZE_PLS
T3(IB){
#ifndef ENABLE_ADVANCED_BLEND
P0=M0(0,0,0,0);
#endif
#ifdef STORE_COLOR_CLEAR
F0(c0,unpackUnorm4x8(v.g9));
#endif
#ifdef SWIZZLE_COLOR_BGRA_TO_RGBA
i f=N0(c0);F0(c0,f.zyxw);
#endif
p3(Q0,v.h9);
#ifdef ENABLE_CLIPPING
a1(w0,0u);
#endif
m3}
#endif
#ifdef RESOLVE_PLS
#ifdef COALESCED_PLS_RESOLVE_AND_TRANSFER
n3(IB)
#else
T3(IB)
#endif
{uint m1=o3(Q0);h Y0=r4(m1&0xffffu);M c1=O0(m1>>16);x0 E=q2(PB,c1);i f=v4(Y0,E,c1 L2 l1);
#ifdef COALESCED_PLS_RESOLVE_AND_TRANSFER
P0=V6(f,E l1);V3
#else
#ifdef ENABLE_ADVANCED_BLEND
A5(f,E l1);
#else
P0=p2(f);
#endif
m3
#endif
}
#endif
#endif
