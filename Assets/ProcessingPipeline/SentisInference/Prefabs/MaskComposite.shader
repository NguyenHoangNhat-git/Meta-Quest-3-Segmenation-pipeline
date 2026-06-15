Shader "CVMQ3/MaskComposite"
{
    Properties
    {
        _MainTex ("Crop Texture", 2D) = "white" {}
        _MaskTex ("Mask Texture", 2D) = "white" {}
        _TintColor ("Tint Color", Color) = (0.2, 0.8, 0.2, 0.5)
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; };
            struct v2f    { float4 pos : SV_POSITION;  float2 uv : TEXCOORD0; };

            sampler2D _MainTex;
            sampler2D _MaskTex;
            fixed4 _TintColor;

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                fixed4 crop = tex2D(_MainTex, i.uv);
                float  mask = tex2D(_MaskTex, i.uv).r;
                crop.rgb = lerp(crop.rgb, _TintColor.rgb, mask * _TintColor.a);
                crop.a = 1.0;
                return crop;
            }
            ENDCG
        }
    }
}