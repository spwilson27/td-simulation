// Split a particle splat into bus sources. a of the splat is heat (1/s); smoke is heat * uSrc.x.

uniform vec4 uSrc;

out vec4 fragColor;
void main()
{
	float heat = texture(sTD2DInputs[0], vUV.st).a;
	fragColor = TDOutputSwizzle(vec4(heat * uSrc.x, heat, 0.0, 0.0));
}
