// Pack a 3D texture into a 2D slice atlas. uAtlas = (cols, rows, volumeW, volumeH), uAtlas2.x = volumeD.
// Pixel shader: TDOutputSwizzle keeps the EXR channels in the same order Movie File In reads back.
uniform vec4 uAtlas;
uniform vec4 uAtlas2;
out vec4 fragColor;
void main()
{
	ivec2 p = ivec2(gl_FragCoord.xy);
	int volW = int(uAtlas.z);
	int volH = int(uAtlas.w);
	int cols = int(uAtlas.x);
	int depth = int(uAtlas2.x);
	ivec2 tile = p / ivec2(volW, volH);
	ivec2 local = p - tile * ivec2(volW, volH);
	int slice = tile.y * cols + tile.x;
	if (slice >= depth || slice < 0)
	{
		fragColor = TDOutputSwizzle(vec4(0.0));
		return;
	}
	fragColor = TDOutputSwizzle(texelFetch(sTD3DInputs[0], ivec3(local, slice), 0));
}
