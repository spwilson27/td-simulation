#include <bus3d_glsl>
#include <liquid_common_glsl>
// Product of two scalar fields on liquid cells. The reduction pass sums this.
// in0, in1 scalars, in2 type.
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	float prod = 0.0;
	if (lqKind(texelFetch(sTD3DInputs[2], c, 0).r) == LQ_LIQUID)
		prod = texelFetch(sTD3DInputs[0], c, 0).r * texelFetch(sTD3DInputs[1], c, 0).r;
	TDImageStoreOutput(0, c, vec4(prod, 0.0, 0.0, 1.0));
}
