// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Custom/DepthGrayscale" {
SubShader {
Tags { "RenderType"="Opaque" }

Pass{
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"

sampler2D _CameraDepthTexture;
 
struct v2f
{
    float4 vertex : SV_POSITION;
    float4 scrPos : TEXCOORD1;
};

v2f vert (appdata_base v)
{
    v2f o;
    o.vertex = UnityObjectToClipPos(v.vertex);
    o.scrPos = ComputeScreenPos(o.vertex);
    return o;
}
 
fixed4 frag (v2f i) : SV_Target
{
    float depth = (tex2Dproj(_CameraDepthTexture, UNITY_PROJ_COORD(i.scrPos)));
    depth = Linear01Depth(depth);
    return depth;
}
ENDCG
}
}
FallBack "Diffuse"
}