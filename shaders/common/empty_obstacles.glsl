#include <bus_glsl>
// Obstacle field with nothing in it: far away, not moving. Input 0 of the obstacle merge.
out vec4 fragColor;
void main()
{
	fragColor = TDOutputSwizzle(vec4(BUS_EMPTY_DISTANCE, 0.0, 0.0, 0.0));
}
