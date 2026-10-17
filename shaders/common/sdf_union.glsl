// Bus merge for obstacles: per texel keep the nearest obstacle (smallest signed distance) and its velocity.
// Input 0 is the empty field (distance BUS_EMPTY_DISTANCE).
out vec4 fragColor;
void main()
{
	vec4 best = texture(sTD2DInputs[0], vUV.st);
	for (int i = 1; i < TD_NUM_2D_INPUTS; i++)
	{
		vec4 o = texture(sTD2DInputs[i], vUV.st);
		if (o.r < best.r) best = o;
	}
	fragColor = TDOutputSwizzle(best);
}
