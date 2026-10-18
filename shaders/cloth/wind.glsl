// Uniform velocity volume. The cloth samples this when it is not reading the bus.
uniform vec4 uWind;
// td: output3d
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 res = ivec3(int(uTDOutputInfo.res.z), int(uTDOutputInfo.res.w), int(uTDOutputInfo.depth.y + 0.5));
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res)))
		return;
	TDImageStoreOutput(0, c, vec4(uWind.xyz, 0.0));
}
