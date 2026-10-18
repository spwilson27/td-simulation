#include <bus3d_glsl>
// Kinematic sphere the liquid (and the bus) can see. uSphere.w is the radius; 0 disables it.
// The centre rides uPath: base xyz plus amplitude * sin(time) on x, velocity amplitude * cos(time).
// td: output3d
uniform vec4 uSphere;
uniform vec4 uPath;
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	if (uSphere.w <= 0.0)
	{
		TDImageStoreOutput(0, c, vec4(BUS_EMPTY_DISTANCE, 0.0, 0.0, 0.0));
		return;
	}
	vec3 centre = uSphere.xyz + vec3(uPath.x * sin(BUS_TIME), 0.0, uPath.y * sin(BUS_TIME + 1.3));
	vec3 velocity = vec3(uPath.x * cos(BUS_TIME), 0.0, uPath.y * cos(BUS_TIME + 1.3));
	vec3 p = busUVToWorld(busUV(c));
	float sd = length(p - centre) - uSphere.w;
	TDImageStoreOutput(0, c, vec4(sd, velocity));
}
