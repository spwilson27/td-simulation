// Split a particle splat into bus forces (xyz acceleration).

uniform vec4 uUnused;

out vec4 fragColor;
void main()
{
	vec4 s = texture(sTD2DInputs[0], vUV.st);
	// The integer splat can leave a 1e-9 residual that differs after a resume. Snap it off.
	vec3 f = round(s.rgb * 1e6) / 1e6;
	fragColor = TDOutputSwizzle(vec4(f, 0.0));
}
