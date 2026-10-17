#include <bus_glsl>
// One Jacobi iteration of the pressure Poisson equation per pass (run with the GLSL TOP's Num Passes: pass k reads
// pass k-1 through input 0; pass 0 reads last step's pressure as a warm start). Neighbours inside obstacles mirror
// the centre value (no flow through walls), as do the domain sides.
// in0: pressure, in1: divergence, in2: obstacles

float pAt(vec2 uv, float pc)
{
	if (busOutside(uv))
		return pc;
	return busSolid(texture(sTD2DInputs[2], uv)) ? pc : texture(sTD2DInputs[0], uv).r;
}

out vec4 fragColor;
void main()
{
	vec2 uv = vUV.st;
	vec2 t = busTexel();
	float h = busCell();
	float pc = texture(sTD2DInputs[0], uv).r;
	float sum = pAt(uv + vec2(t.x, 0), pc) + pAt(uv - vec2(t.x, 0), pc)
	          + pAt(uv + vec2(0, t.y), pc) + pAt(uv - vec2(0, t.y), pc);
	float div = texture(sTD2DInputs[1], uv).r;
	fragColor = TDOutputSwizzle(vec4((sum - h * h * div) * 0.25, 0.0, 0.0, 1.0));
}
