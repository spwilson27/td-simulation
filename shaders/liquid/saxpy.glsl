#include <liquid_common_glsl>
#include <bus3d_glsl>
// out = uCoef.x * in0 + uCoef.y * in1 on liquid cells, otherwise 0.
// in0, in1 scalar fields, in2 type.
// td: output3d
uniform vec4 uCoef;
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
	float a = texelFetch(sTD3DInputs[0], c, 0).r;
	float b = texelFetch(sTD3DInputs[1], c, 0).r;
	TDImageStoreOutput(0, c, vec4(uCoef.x * a + uCoef.y * b, 0.0, 0.0, 1.0));
}
