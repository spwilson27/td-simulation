#include <bus_glsl>
// Semi-Lagrangian advection (RK2 backtrace) of any field by the velocity field, with per-channel decay and the
// absorbing band at open domain sides.
// in0: field q, in1: velocity, in2: obstacles
uniform vec4 uDecay;      // per-channel decay rate (1/s)
uniform vec4 uAdvect;     // x: 1 when q is the velocity field (solid cells take the obstacle velocity)

out vec4 fragColor;
void main()
{
	vec2 uv = vUV.st;
	vec4 obstacle = texture(sTD2DInputs[2], uv);
	vec2 back = busBacktrace(sTD2DInputs[1], uv, BUS_DT);
	vec4 q = texture(sTD2DInputs[0], back) * exp(-(uDecay + busSponge(uv) * uSponge.y) * BUS_DT);
	if (busSolid(obstacle))
		q = uAdvect.x > 0.5 ? vec4(obstacle.gb, 0.0, 0.0) : vec4(0.0);
	fragColor = TDOutputSwizzle(q);
}
