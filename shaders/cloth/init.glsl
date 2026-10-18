#include <cloth_common_glsl>
// Rest pose. w is inverse mass; 0 pins the vertex. uPin.x is 0 left edge, 1 two corners, 2 free, 3 top edge.
// uPlane.x is 0 for an xy flag and 1 for an xz drape.
uniform vec4 uOrigin;
uniform vec4 uSize;
uniform vec4 uPin;
uniform vec4 uMass;
uniform vec4 uPlane;
layout(local_size_x = 8, local_size_y = 8) in;

void main()
{
	ivec2 size = ivec2(uTDOutputInfo.res.zw);
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	if (coord.x >= size.x || coord.y >= size.y)
		return;
	float dx = uSize.x / float(max(size.x - 1, 1));
	float dy = uSize.y / float(max(size.y - 1, 1));
	int i = coord.x;
	int j = coord.y;
	vec3 p = uPlane.x < 0.5
		? vec3(uOrigin.x + float(i) * dx, uOrigin.y - float(j) * dy, uOrigin.z)
		: vec3(uOrigin.x + float(i) * dx, uOrigin.y, uOrigin.z + float(j) * dy);
	float inv = 1.0 / max(uMass.x * dx * dy, 1e-8);
	int mode = int(uPin.x + 0.5);
	if (mode == 0 && i == 0)
		inv = 0.0;
	if (mode == 1 && j == 0 && (i == 0 || i == size.x - 1))
		inv = 0.0;
	if (mode == 3 && j == 0)
		inv = 0.0;
	TDImageStoreOutput(0, ivec3(coord, 0), vec4(p, inv));
}
