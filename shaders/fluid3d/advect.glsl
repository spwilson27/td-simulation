#include <bus3d_glsl>
// RK2 semi-Lagrangian advection. in0 field, in1 velocity, in2 obstacles.
// uDecay: per-channel decay (1/s). uAdvect.x > 0.5 when the field is velocity (solids take obstacle velocity).
uniform vec4 uDecay;
uniform vec4 uAdvect;
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	vec3 uv = busUV(c);
	vec4 obstacle = texture(sTD3DInputs[2], uv);
	vec3 back = busBacktrace(sTD3DInputs[1], uv, BUS_DT);
	vec4 q = texture(sTD3DInputs[0], back) * exp(-(uDecay + busSponge(uv) * uSponge.y) * BUS_DT);
	if (busSolid(obstacle))
		q = uAdvect.x > 0.5 ? vec4(busObstacleVel(obstacle), 0.0) : vec4(0.0);
	TDImageStoreOutput(0, c, q);
}
