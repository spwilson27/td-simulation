// Probe: with N passes, does pass k read the output of pass k-1 through input 0, and is uTDPass set?
out vec4 fragColor;
void main()
{
	vec4 prev = texture(sTD2DInputs[0], vUV.st);
	fragColor = TDOutputSwizzle(prev + vec4(1.0, float(uTDPass), 0.0, 0.0));
}
