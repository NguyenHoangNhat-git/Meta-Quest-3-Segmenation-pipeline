Shader "CVMQ3/SegmentationOverlay"
{
    Properties
    {
        _MainTex ("Overlay Texture", 2D) = "white" {}
        _MaskTex ("Mask Texture", 2D) = "white" {}
        _Color ("Tint Color", Color) = (0.2, 0.8, 0.2, 0.5)
    }

    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" }
        LOD 100

        ZWrite Off
        ZTest LEqual
        Blend SrcAlpha OneMinusSrcAlpha
        Cull Off  // render both sides

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            sampler2D _MainTex;
            sampler2D _MaskTex;
            fixed4 _Color;

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                // _MaskTex is 160x160 covering full 640x640 input
                // UVs are already in bbox-normalized space — sample directly
                float mask = tex2D(_MaskTex, i.uv).r;
                clip(mask - 0.5);

                fixed4 col = tex2D(_MainTex, i.uv);
                col.rgb = lerp(col.rgb, _Color.rgb, _Color.a);
                col.a = mask * _Color.a;
                return col;
            }
            ENDCG
        }
    }
}