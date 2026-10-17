#include <bus_glsl>
// Subtract the pressure gradient to make the velocity divergence-free; solid cells take the obstacle velocity.
// in0: velocity, in1: pressure, in2: obstacles
uniform vec4 uLimit;      // x: maximum speed (world units / s), keeps runaway cells bounded

float pAt(vec2 uv, float pc)
{
	if (busOutside(uv))
		return pc;
	return busSolid(texture(sTD2DInputs[2], uv)) ? pc : texture(sTD2DInputs[1], uv).r;
}

out vec4 fragColor;
void main()
{
	vec2 uv = vUV.st;
	vec4 obstacle = texture(sTD2DInputs[2], uv);
	if (busSolid(obstacle))
	{
		fragColor = TDOutputSwizzle(vec4(obstacle.gb, 0.0, 0.0));
		return;
	}
	vec2 t = busTexel();
	float pc = texture(sTD2DInputs[1], uv).r;
	vec2 grad = vec2(pAt(uv + vec2(t.x, 0), pc) - pAt(uv - vec2(t.x, 0), pc),
	                 pAt(uv + vec2(0, t.y), pc) - pAt(uv - vec2(0, t.y), pc)) / (2.0 * busCell());
	vec2 v = texture(sTD2DInputs[0], uv).xy - grad;
	float speed = length(v);
	if (speed > uLimit.x)
		v *= uLimit.x / speed;
	fragColor = TDOutputSwizzle(vec4(v, 0.0, 0.0));
}
