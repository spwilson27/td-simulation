in vec4 vColor;
out vec4 fragColor;

void main()
{
	vec2 q = TDPointCoord() - vec2(0.5);
	float r2 = dot(q, q);
	if (r2 > 0.25)
		discard;
	float fall = smoothstep(0.25, 0.02, r2);
	vec3 rgb = vColor.rgb * fall * vColor.a;
	fragColor = vec4(rgb, fall * vColor.a);
}
