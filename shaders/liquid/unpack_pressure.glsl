#include <bus3d_glsl>
// Expand a 2D packing of a pressure volume. Row = z * height + y, column = x.
// The pack is written from numpyArray, whose row 0 is the top of the texture, matching texelFetch.
// in0 packed pressure.
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	ivec3 res = ivec3(busRes());
	int row = c.z * res.y + c.y;
	ivec2 size = textureSize(sTD2DInputs[0], 0);
	float p = 0.0;
	if (c.x < size.x && row >= 0 && row < size.y)
		p = texelFetch(sTD2DInputs[0], ivec2(c.x, row), 0).r;
	TDImageStoreOutput(0, c, vec4(p, 0.0, 0.0, 1.0));
}
