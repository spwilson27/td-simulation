// 3D deposit. in0/in1 particle textures (2D). Bus velocity is sTD3DInputs[0].

uniform vec4 uCouple;
uniform vec4 uPhys;
uniform vec4 uField;
uniform vec4 uAnalytic;
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uScale;     // x cell size, y particle mass, z fluid density

vec3 toUV(vec3 p)
{
	return clamp((p - uBMin.xyz) / max(uBMax.xyz - uBMin.xyz, vec3(1e-6)), vec3(0.0), vec3(1.0));
}

vec3 fieldVelocity(vec3 p)
{
	if (uField.x > 1.5)
	{
		vec2 r = p.xy - uAnalytic.xy;
		return vec3(-uAnalytic.w * r.y, uAnalytic.w * r.x, 0.0);
	}
	if (uField.x > 0.5)
		return uAnalytic.xyz;
	return texture(sTD3DInputs[0], toUV(p)).xyz;
}

out vec4 fragColor;
void main()
{
	ivec2 coord = ivec2(gl_FragCoord.xy);
	vec3 pos = texelFetch(sTD2DInputs[0], coord, 0).xyz;
	vec4 pv = texelFetch(sTD2DInputs[1], coord, 0);
	if (pv.w <= 0.0 || uCouple.x == 0.0 && uCouple.y == 0.0)
	{
		fragColor = TDOutputSwizzle(vec4(0.0));
		return;
	}
	float h = max(uScale.x, 1e-6);
	float ratio = (uScale.y * 1e-9) / max(uScale.z * h * h * h, 1e-12);
	float gain = uCouple.x * ratio;
	vec3 force = gain * uPhys.x * (pv.xyz - fieldVelocity(pos));
	fragColor = TDOutputSwizzle(vec4(force, uCouple.y * ratio));
}
