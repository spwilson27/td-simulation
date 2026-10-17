#include <bus3d_glsl>
// Add the trilinear coarse correction onto the finer pressure. in0 fine pressure, in1 coarse correction.
// uScale.x is 1 for a real V-cycle. A gain near 0 leaves the smoother and does not remove smooth divergence.
uniform vec4 uScale; // x: correction gain
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	vec3 uv = busUV(c);
	// uScale.x damps the coarse correction. 1 is a full V-cycle; 0 is smoothing only.
	float corr = texture(sTD3DInputs[1], uv).r * uScale.x;
	float p = texelFetch(sTD3DInputs[0], c, 0).r + corr;
	TDImageStoreOutput(0, c, vec4(p, 0.0, 0.0, 1.0));
}
