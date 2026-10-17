// The sprite pass is additive, so its alpha is coverage. The saved picture is opaque.

out vec4 fragColor;
void main()
{
	vec4 c = texture(sTD2DInputs[0], vUV.st);
	fragColor = TDOutputSwizzle(vec4(max(c.rgb, vec3(0.0)), 1.0));
}
