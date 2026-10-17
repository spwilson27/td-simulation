// Scatter a cropped volume back into the full grid. Cells outside the crop are 0.
// uOrigin = (x0, y0, z0), uCrop = (cropW, cropH, cropD). in0 is the cropped volume.
uniform vec4 uOrigin;
uniform vec4 uCrop;
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 full = ivec3(uTDOutputInfo.res.zw, uTDOutputInfo.depth.y);
	if (any(greaterThanEqual(c, full)))
		return;
	ivec3 local = c - ivec3(uOrigin.xyz);
	ivec3 crop = ivec3(uCrop.xyz);
	if (any(lessThan(local, ivec3(0))) || any(greaterThanEqual(local, crop)))
	{
		TDImageStoreOutput(0, c, vec4(0.0));
		return;
	}
	TDImageStoreOutput(0, c, texelFetch(sTD3DInputs[0], local, 0));
}
