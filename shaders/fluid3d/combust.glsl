#include <bus3d_glsl>
// Sources and combustion. Same model as the 2D solver.
// in0 scalars, in1 sources, in2 obstacles.
uniform vec4 uBurn;    // x rate, y heat per fuel, z smoke per fuel, w ignition
uniform vec4 uThermo;  // x cooling, y expansion per fuel, z max temperature
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	vec3 uv = busUV(c);
	float dt = BUS_DT;
	vec4 s = texelFetch(sTD3DInputs[0], c, 0);
	vec4 src = texture(sTD3DInputs[1], uv);
	s.rgb += src.rgb * dt;
	float burnFrac = s.g > uBurn.w ? 1.0 - exp(-uBurn.x * dt) : 0.0;
	float burned = s.b * burnFrac;
	s.b -= burned;
	s.g += burned * uBurn.y;
	s.r += burned * uBurn.z;
	s.g = min(s.g * exp(-uThermo.x * dt), uThermo.z);
	s.a = burned / max(dt, 1e-6) * uThermo.y + src.a;
	s.rgb = max(s.rgb, vec3(0.0));
	if (busSolid(texture(sTD3DInputs[2], uv)))
		s = vec4(0.0);
	TDImageStoreOutput(0, c, s);
}
