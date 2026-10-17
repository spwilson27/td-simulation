// Probe: pixel shader reads one slice of a 3D input with texelFetch.
uniform float uSlice;
out vec4 fragColor;
void main()
{
	ivec3 c = ivec3(gl_FragCoord.xy, int(uSlice));
	fragColor = TDOutputSwizzle(texelFetch(sTD3DInputs[0], c, 0));
}
