// mix(in0, in1, uWeight.x). Both inputs are full-grid volumes.
uniform vec4 uWeight;
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 full = ivec3(uTDOutputInfo.res.zw, uTDOutputInfo.depth.y);
	if (any(greaterThanEqual(c, full)))
		return;
	vec4 a = texelFetch(sTD3DInputs[0], c, 0);
	vec4 b = texelFetch(sTD3DInputs[1], c, 0);
	TDImageStoreOutput(0, c, mix(a, b, clamp(uWeight.x, 0.0, 1.0)));
}
