#include <cloth_common_glsl>
// Hash volume laid into a 2D image so a locked null can hold it. Row = y + z * H.
uniform vec4 uHash;
layout(local_size_x = 8, local_size_y = 8) in;

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	int h = max(int(uHash.x + 0.5), 1);
	int y = coord.y - (coord.y / h) * h;
	int z = coord.y / h;
	vec4 bin = vec4(0.0);
	if (coord.x < h && z < h)
		bin = texelFetch(sTD3DInputs[0], ivec3(coord.x, y, z), 0);
	TDImageStoreOutput(0, ivec3(coord, 0), bin);
}
