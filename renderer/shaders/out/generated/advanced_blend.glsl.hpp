#pragma once

#include "advanced_blend.exports.h"

namespace rive {
namespace pls {
namespace glsl {
const char advanced_blend[] = R"===(#ifdef MC
layout(
#ifdef JB
blend_support_all_equations
#else
blend_support_multiply,blend_support_screen,blend_support_overlay,blend_support_darken,blend_support_lighten,blend_support_colordodge,blend_support_colorburn,blend_support_hardlight,blend_support_softlight,blend_support_difference,blend_support_exclusion
#endif
)out;
#endif
#ifdef AB
#ifdef JB
h f5(O e0){return min(min(e0.x,e0.y),e0.z);}h y6(O e0){return max(max(e0.x,e0.y),e0.z);}h g5(O e0){return dot(e0,j0(.30,.59,.11));}h z6(O e0){return y6(e0)-f5(e0);}O D8(O f){h h2=g5(f);h A6=f5(f);h B6=y6(f);if(A6<.0)f=h2+((f-h2)*h2)/(h2-A6);if(B6>1.)f=h2+((f-h2)*(1.-h2))/(B6-h2);return f;}O h5(O i3,O i5){h E8=g5(i3);h F8=g5(i5);h j5=F8-E8;O f=i3+j0(j5,j5,j5);return D8(f);}O C6(O i3,O G8,O i5){h H8=f5(i3);h D6=z6(i3);h I8=z6(G8);O f;if(D6>.0){f=(i3-H8)*I8/D6;}else{f=j0(0,0,0);}return h5(f,i5);}
#endif
#ifdef JB
i E3(i q,i n,M E6)
#else
i F3(i q,i n,M E6)
#endif
{O f0=j0(0,0,0);switch(E6){case J8:f0=q.xyz*n.xyz;break;case K8:f0=q.xyz+n.xyz-q.xyz*n.xyz;break;case L8:{for(int k=0;k<3;++k){if(n[k]<=.5)f0[k]=2.*q[k]*n[k];else f0[k]=1.-2.*(1.-q[k])*(1.-n[k]);}break;}case M8:f0=min(q.xyz,n.xyz);break;case N8:f0=max(q.xyz,n.xyz);break;case O8:f0=mix(min(n.xyz/(1.-q.xyz),j0(1,1,1)),j0(0,0,0),lessThanEqual(n.xyz,j0(0,0,0)));break;case P8:f0=mix(1.-min((1.-n.xyz)/q.xyz,1.),j0(1,1,1),greaterThanEqual(n.xyz,j0(1,1,1)));break;case Q8:{for(int k=0;k<3;++k){if(q[k]<=.5)f0[k]=2.*q[k]*n[k];else f0[k]=1.-2.*(1.-q[k])*(1.-n[k]);}break;}case R8:{for(int k=0;k<3;++k){if(q[k]<=0.5)f0[k]=n[k]-(1.-2.*q[k])*n[k]*(1.-n[k]);else if(n[k]<=.25)f0[k]=n[k]+(2.*q[k]-1.)*n[k]*((16.*n[k]-12.)*n[k]+3.);else f0[k]=n[k]+(2.*q[k]-1.)*(sqrt(n[k])-n[k]);}break;}case S8:f0=abs(n.xyz-q.xyz);break;case T8:f0=q.xyz+n.xyz-2.*q.xyz*n.xyz;break;
#ifdef JB
case U8:q.xyz=clamp(q.xyz,j0(0,0,0),j0(1,1,1));f0=C6(q.xyz,n.xyz,n.xyz);break;case V8:q.xyz=clamp(q.xyz,j0(0,0,0),j0(1,1,1));f0=C6(n.xyz,q.xyz,n.xyz);break;case W8:q.xyz=clamp(q.xyz,j0(0,0,0),j0(1,1,1));f0=h5(q.xyz,n.xyz);break;case X8:q.xyz=clamp(q.xyz,j0(0,0,0),j0(1,1,1));f0=h5(n.xyz,q.xyz);break;
#endif
}O G3=j0(q.w*n.w,q.w*(1.-n.w),(1.-q.w)*n.w);return h0(k5(f0,1,q.xyz,1,n.xyz,1),G3);}
#endif
)===";
} // namespace glsl
} // namespace pls
} // namespace rive