#include <bus_glsl>
// Velocity divergence minus the combustion expansion. Neighbours inside obstacles contribute the obstacle velocity.
// in0: velocity, in1: obstacles, in2: scalars (a = expansion rate)

vec2 velAt(vec2 uv)
{
	if (busOutside(uv))
		return vec2(0.0);
	vec4 o = texture(sTD2DInputs[1], uv);
	return busSolid(o) ? o.gb : texture(sTD2DInputs[0], uv).xy;
}

out vec4 fragColor;
void main()
{
	vec2 uv = vUV.st;
	vec2 t = busTexel();
	float div = (velAt(uv + vec2(t.x, 0)).x - velAt(uv - vec2(t.x, 0)).x
	           + velAt(uv + vec2(0, t.y)).y - velAt(uv - vec2(0, t.y)).y) / (2.0 * busCell());
	div -= texture(sTD2DInputs[2], uv).a;
	fragColor = TDOutputSwizzle(vec4(div, 0.0, 0.0, 1.0));
}
