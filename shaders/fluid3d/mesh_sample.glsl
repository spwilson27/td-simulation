#include <bus3d_glsl>
// Rigid transform of a baked local SDF into a bus obstacle (distance + velocity).
// in0: local SDF volume. uCenter xyz is the current world centre.
// uYaw.x angle (rad), uYaw.y angular speed about Y, uYaw.z uniform scale, uYaw.w enable.
// uVel xyz is the centre's linear velocity. uSdf0/uSdf1 are the local volume bounds.
uniform vec4 uCenter;
uniform vec4 uYaw;
uniform vec4 uVel;
uniform vec4 uSdf0;
uniform vec4 uSdf1;
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

vec3 rotY(vec3 p, float a)
{
	float s = sin(a), c = cos(a);
	return vec3(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}

void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	if (uYaw.w < 0.5)
	{
		TDImageStoreOutput(0, c, vec4(BUS_EMPTY_DISTANCE, 0.0, 0.0, 0.0));
		return;
	}
	vec3 world = busUVToWorld(busUV(c));
	vec3 d = world - uCenter.xyz;
	float scale = max(abs(uYaw.z), 1e-4);
	vec3 local = rotY(d, -uYaw.x) / scale;
	vec3 uvw = (local - uSdf0.xyz) / max(uSdf1.xyz - uSdf0.xyz, vec3(1e-6));
	float sdf;
	if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0))))
		sdf = BUS_EMPTY_DISTANCE;
	else
		sdf = texture(sTD3DInputs[0], clamp(uvw, vec3(0.0), vec3(1.0))).r * scale;
	vec3 vel = uVel.xyz + cross(vec3(0.0, uYaw.y, 0.0), d);
	TDImageStoreOutput(0, c, vec4(sdf, vel));
}
