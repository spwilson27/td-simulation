#include <bus_glsl>
// Scalar sources and combustion. Fuel above the ignition temperature burns at a fixed rate, releasing heat, smoke
// and expansion (a divergence source the pressure solve turns into outward flow). Temperature cools exponentially.
// in0: scalars (advected), in1: sources, in2: obstacles
uniform vec4 uBurn;       // x burn rate (1/s), y heat per fuel, z smoke per fuel, w ignition temperature
uniform vec4 uThermo;     // x cooling rate (1/s), y expansion per fuel burned, z max temperature

out vec4 fragColor;
void main()
{
	vec2 uv = vUV.st;
	float dt = BUS_DT;
	vec4 s = texture(sTD2DInputs[0], uv);
	vec4 src = texture(sTD2DInputs[1], uv);
	s.rgb += src.rgb * dt;

	float burnFrac = s.g > uBurn.w ? 1.0 - exp(-uBurn.x * dt) : 0.0;
	float burned = s.b * burnFrac;
	s.b -= burned;
	s.g += burned * uBurn.y;
	s.r += burned * uBurn.z;
	s.g = min(s.g * exp(-uThermo.x * dt), uThermo.z);
	s.a = burned / max(dt, 1e-6) * uThermo.y + src.a;
	s.rgb = max(s.rgb, vec3(0.0));

	if (busSolid(texture(sTD2DInputs[2], uv)))
		s = vec4(0.0);
	fragColor = TDOutputSwizzle(s);
}
