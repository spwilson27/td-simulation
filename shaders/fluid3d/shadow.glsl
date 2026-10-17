#include <bus3d_glsl>
// Transmittance along uLight.xyz through smoke density, stored per voxel so the raymarcher can sample it.
// in0 scalars. uLook.x is the smoke extinction coefficient (same as the renderer).
uniform vec4 uLight;
uniform vec4 uLook;
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	vec3 dir = normalize(uLight.xyz);
	vec3 uv = busUV(c);
	const float steps = 8.0;
	vec3 stepUv = dir * (length(busSize()) / steps) / busSize();
	float trans = 1.0;
	for (int i = 0; i < 8; i++)
	{
		uv += stepUv;
		if (busOutside(uv))
			break;
		float dens = max(texture(sTD3DInputs[0], uv).r, 0.0);
		trans *= exp(-dens * uLook.x * length(stepUv * busSize()));
	}
	TDImageStoreOutput(0, c, vec4(trans, 0.0, 0.0, 1.0));
}
