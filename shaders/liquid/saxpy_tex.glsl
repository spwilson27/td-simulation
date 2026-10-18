#include <liquid_common_glsl>
#include <bus3d_glsl>
// out = coef.r * in0 + uSign.x * coef.g * in1 on liquid cells, otherwise 0.
// in0 and in1 are the fields, in2 is the cell type, in3 is a 1x1x1 coef texture.
// The coef comes from the GPU (alpha or beta). uSign is a constant on the node.
// td: output3d
uniform vec4 uSign;
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	if (lqKind(texelFetch(sTD3DInputs[2], c, 0).r) != LQ_LIQUID)
	{
		TDImageStoreOutput(0, c, vec4(0.0));
		return;
	}
	vec4 coef = texelFetch(sTD3DInputs[3], ivec3(0), 0);
	float a = texelFetch(sTD3DInputs[0], c, 0).r;
	float b = texelFetch(sTD3DInputs[1], c, 0).r;
	TDImageStoreOutput(0, c, vec4(coef.r * a + uSign.x * coef.g * b, 0.0, 0.0, 1.0));
}
