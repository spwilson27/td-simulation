// Bus merge: sum of every input (input 0 is a zero texture that fixes resolution and format).
out vec4 fragColor;
void main()
{
	vec4 acc = vec4(0.0);
	for (int i = 0; i < TD_NUM_2D_INPUTS; i++)
		acc += texture(sTD2DInputs[i], vUV.st);
	fragColor = TDOutputSwizzle(acc);
}
