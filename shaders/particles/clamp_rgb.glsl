// Point renders leave garbage in untouched float pixels. Clamp before additive compositing.

out vec4 fragColor;
void main()
{
	vec4 c = texture(sTD2DInputs[0], vUV.st);
	fragColor = TDOutputSwizzle(vec4(max(c.rgb, vec3(0.0)), max(c.a, 0.0)));
}
