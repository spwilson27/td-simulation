#include <bus_glsl>
// Obstacle publisher: a moving disc. Writes signed distance and the disc velocity everywhere (the velocity matters
// only where the distance is negative, but keeping it smooth helps sources that sample near the surface).
uniform vec4 uDisc;         // xy world centre, z radius
uniform vec4 uDiscVel;      // xy velocity (world / s)

out vec4 fragColor;
void main()
{
	vec2 p = busUVToWorld(vUV.st);
	fragColor = TDOutputSwizzle(vec4(sdCircle(p, uDisc.xy, uDisc.z), uDiscVel.xy, 0.0));
}
