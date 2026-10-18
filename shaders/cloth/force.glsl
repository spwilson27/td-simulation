#include <cloth_common_glsl>
// Reaction of the cloth force on the fluid. Particles are visited in sorted index order.
// 2D in0 packed bins, in1 sorted keys, in2 aero force, in3 positions.
// uScale.x converts newtons to an acceleration added into the bus forces field.
uniform vec4 uBMin;
uniform vec4 uBMax;
uniform vec4 uHash;
uniform vec4 uPart;
uniform vec4 uScale;
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

vec4 packedBin(ivec3 c, int h)
{
	return texelFetch(sTD2DInputs[0], ivec2(c.x, c.y + c.z * h), 0);
}

ivec3 busCell(vec3 p, ivec3 res)
{
	vec3 span = max(uBMax.xyz - uBMin.xyz, vec3(1e-6));
	ivec3 c = ivec3(floor(((p - uBMin.xyz) / span) * vec3(res)));
	return clamp(c, ivec3(0), res - ivec3(1));
}

void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 res = ivec3(int(uTDOutputInfo.res.z), int(uTDOutputInfo.res.w), int(uTDOutputInfo.depth.y + 0.5));
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res)))
		return;
	int h = max(int(uHash.x + 0.5), 1);
	int width = int(uPart.x + 0.5);
	vec3 span = max(uBMax.xyz - uBMin.xyz, vec3(1e-6));
	vec3 centre = uBMin.xyz + (vec3(c) + 0.5) * span / vec3(res);
	ivec3 cell = clothCell(centre, uBMin.xyz, uBMax.xyz, h);
	vec4 bin = packedBin(cell, h);
	int start = int(bin.r + 0.5);
	int n = min(int(bin.g + 0.5), 16);
	vec3 acc = vec3(0.0);
	for (int k = 0; k < 16; ++k)
	{
		if (k >= n)
			break;
		int id = int(texelFetch(sTD2DInputs[1], clothTexel(start + k, width), 0).g + 0.5);
		ivec2 pix = clothTexel(id, width);
		if (busCell(texelFetch(sTD2DInputs[3], pix, 0).xyz, res) != c)
			continue;
		acc += -texelFetch(sTD2DInputs[2], pix, 0).xyz * uScale.x;
	}
	TDImageStoreOutput(0, c, vec4(acc, 0.0));
}
