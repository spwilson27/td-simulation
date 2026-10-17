// Per-particle deposit into the bus, from the snapshotted state (not the feedback loop).
// in0 position+age, in1 velocity+life, in2 bus velocity.
// rgb = acceleration given to the fluid (drag reaction), a = heat injection (1/s).

uniform vec4 uCouple;    // x force scale, y heat per alive particle
uniform vec4 uPhys;      // x drag (1/s)
uniform vec4 uField;
uniform vec4 uAnalytic;
uniform vec4 uBounds;
uniform vec4 uBMin;      // unused in 2D; present so the builder can share the uniform block
uniform vec4 uBMax;
uniform vec4 uScale;     // x cell size, y particle mass, z fluid density

vec2 toUV(vec3 p)
{
	return clamp((p.xy - uBounds.xy) / max(uBounds.zw - uBounds.xy, vec2(1e-6)), vec2(0.0), vec2(1.0));
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
	return vec3(texture(sTD2DInputs[2], toUV(p)).xy, 0.0);
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
	// Drag on the particle is drag * (v_fluid - v_p). The fluid gets the opposite momentum,
	// divided by the mass of fluid in the cell so a crowd of light embers cannot dominate it.
	float h = max(uScale.x, 1e-6);
	// Particlemass is in nanograms so the parameter stays in a range TouchDesigner stores exactly.
	float ratio = (uScale.y * 1e-9) / max(uScale.z * h * h, 1e-12);
	float gain = uCouple.x * ratio;
	vec3 force = gain * uPhys.x * (pv.xyz - fieldVelocity(pos));
	fragColor = TDOutputSwizzle(vec4(force, uCouple.y * ratio));
}
