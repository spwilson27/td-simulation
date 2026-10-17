// Pack a cropped region of a 3D texture into a 2D slice atlas.
// uAtlas = (cols, rows, cropW, cropH), uAtlas2.x = cropD, uOrigin = (x0, y0, z0) in the full volume.
uniform vec4 uAtlas;
uniform vec4 uAtlas2;
uniform vec4 uOrigin;
out vec4 fragColor;
void main()
{
	ivec2 p = ivec2(gl_FragCoord.xy);
	int volW = int(uAtlas.z);
	int volH = int(uAtlas.w);
	int cols = int(uAtlas.x);
	int depth = int(uAtlas2.x);
	ivec2 tile = p / ivec2(max(volW, 1), max(volH, 1));
	ivec2 local = p - tile * ivec2(volW, volH);
	int slice = tile.y * cols + tile.x;
	if (slice >= depth || slice < 0 || local.x >= volW || local.y >= volH)
	{
		fragColor = TDOutputSwizzle(vec4(0.0));
		return;
	}
	ivec3 c = ivec3(local, slice) + ivec3(uOrigin.xyz);
	ivec3 size = textureSize(sTD3DInputs[0], 0);
	if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, size)))
	{
		fragColor = TDOutputSwizzle(vec4(0.0));
		return;
	}
	fragColor = TDOutputSwizzle(texelFetch(sTD3DInputs[0], c, 0));
}
