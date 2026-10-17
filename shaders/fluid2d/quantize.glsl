// Snap to a fixed grid so two coupled bakes land on the same bits. The GPU's last-bit
// noise is smaller than this step; without it, EXR bytes of an otherwise identical run differ.
uniform vec4 uStep; // x reciprocal step (values are rounded to 1/x)

out vec4 fragColor;
void main()
{
	vec4 c = texture(sTD2DInputs[0], vUV.st);
	float s = max(uStep.x, 1.0);
	fragColor = TDOutputSwizzle(round(c * s) / s);
}
