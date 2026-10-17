// Additive deposit. The vertex shader already snapped the point to a pixel centre.

in vec4 vReact;
out vec4 fragColor;

void main()
{
	fragColor = vReact;
}
