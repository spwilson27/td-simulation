// Unpack a 2D slice atlas into a 3D texture. texture() (nearest) rather than texelFetch so TD's channel
// swizzle matches the pixel shader that wrote the EXR. uAtlas = (cols, rows, volumeW, volumeH), uAtlas2.x = depth.
uniform vec4 uAtlas;
uniform vec4 uAtlas2;
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = ivec3(gl_GlobalInvocationID.xyz);
	int volW = int(uAtlas.z);
	int volH = int(uAtlas.w);
	int cols = int(uAtlas.x);
	int depth = int(uAtlas2.x);
	if (c.x >= volW || c.y >= volH || c.z >= depth)
		return;
	int col = c.z % cols;
	int row = c.z / cols;
	ivec2 p = ivec2(col * volW + c.x, row * volH + c.y);
	ivec2 size = textureSize(sTD2DInputs[0], 0);
	vec2 uv = (vec2(p) + 0.5) / vec2(size);
	TDImageStoreOutput(0, c, texture(sTD2DInputs[0], uv));
}
