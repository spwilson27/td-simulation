// Per-channel gain, e.g. to republish cached smoke/heat as sources for another simulation.
uniform vec4 uGain;
out vec4 fragColor;
void main()
{
	fragColor = TDOutputSwizzle(texture(sTD2DInputs[0], vUV.st) * uGain);
}
