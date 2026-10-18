// Side view of the liquid phi: the 2D preview, and the small picture stored between bake keyframes.
// in0 phi volume.
uniform vec4 uUnused;

out vec4 fragColor;
void main()
{
	vec2 uv = vUV.st;
	ivec3 res = textureSize(sTD3DInputs[0], 0);
	int z = res.z / 2;
	ivec2 px = ivec2(uv * vec2(res.xy));
	px = clamp(px, ivec2(0), res.xy - ivec2(1));
	float phi = texelFetch(sTD3DInputs[0], ivec3(px, z), 0).r;
	vec3 bg = vec3(0.05, 0.06, 0.08);
	vec3 col = bg;
	if (phi < 1.0)
	{
		float band = clamp(0.5 - phi * 2.0, 0.0, 1.0);
		col = mix(vec3(0.15, 0.35, 0.55), vec3(0.02, 0.12, 0.22), band);
	}
	if (phi < 0.0)
		col = mix(vec3(0.25, 0.55, 0.7), vec3(0.03, 0.15, 0.28), clamp(-phi * 8.0, 0.0, 1.0));
	fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
