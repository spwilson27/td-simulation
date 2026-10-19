// Split a particle splat into bus forces (xyz acceleration).

uniform vec4 uUnused;

out vec4 fragColor;
void main()
{
	vec3 f = texelFetch(sTD2DInputs[0], ivec2(gl_FragCoord.xy), 0).rgb;
	fragColor = TDOutputSwizzle(vec4(f, 0.0));
}
