// Split a particle splat into bus sources. a of the splat is heat (1/s); smoke is heat * uSrc.x.

uniform vec4 uSrc;

out vec4 fragColor;
void main()
{
	float heat = texelFetch(sTD2DInputs[0], ivec2(gl_FragCoord.xy), 0).a;
	fragColor = TDOutputSwizzle(vec4(heat * uSrc.x, heat, 0.0, 0.0));
}
