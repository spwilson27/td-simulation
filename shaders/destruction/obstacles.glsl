#include <bus3d_glsl>
// Moving convex pieces as oriented boxes. in0 body state, in1 extents, in2 bin list.
// uCount: x bodies, y grid, z pixels per bin, w flip state-texture rows (texelFetch vs numpy).
uniform vec4 uCount;

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

int flipY(int y, int height)
{
	return uCount.w > 0.5 ? (height - 1 - y) : y;
}

ivec2 statePix(int id, int quad)
{
	ivec2 size = textureSize(sTD2DInputs[0], 0);
	int x = id % 128;
	int y = id / 128;
	if (quad == 1 || quad == 3) x += 128;
	if (quad == 2 || quad == 3) y += 128;
	return ivec2(x, flipY(y, size.y));
}

ivec2 binPix(int linear)
{
	ivec2 size = textureSize(sTD2DInputs[2], 0);
	int x = linear % max(size.x, 1);
	int y = linear / max(size.x, 1);
	return ivec2(x, flipY(y, size.y));
}

vec3 qrot(vec4 q, vec3 v)
{
	return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

float sdOriented(vec3 p, vec3 center, vec4 quat, vec3 halfExt, vec3 localCenter)
{
	vec3 local = qrot(vec4(-quat.xyz, quat.w), p - center) - localCenter;
	vec3 q = abs(local) - max(halfExt, vec3(1e-4));
	return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

void main()
{
	ivec3 c = busCoord();
	if (!busLive(c))
		return;
	vec3 p = busUVToWorld(busUV(c));
	vec3 size = max(busSize(), vec3(1e-4));
	int grid = max(int(uCount.y + 0.5), 1);
	ivec3 bin = ivec3(clamp((p - uBMin.xyz) / size * float(grid), vec3(0.0), vec3(float(grid) - 1.0)));
	int binId = bin.x + grid * (bin.y + grid * bin.z);
	int stride = max(int(uCount.z + 0.5), 1);
	int start = binId * stride;
	int count = int(texelFetch(sTD2DInputs[2], binPix(start), 0).r + 0.5);
	float best = BUS_EMPTY_DISTANCE;
	vec3 vel = vec3(0.0);
	for (int i = 0; i < 8; i++)
	{
		if (i >= count)
			break;
		int texel = i / 4;
		int chan = i - texel * 4;
		vec4 packed = texelFetch(sTD2DInputs[2], binPix(start + 1 + texel), 0);
		int id = int(packed[chan] + 0.5);
		if (id < 0 || id >= int(uCount.x + 0.5))
			continue;
		vec4 pose = texelFetch(sTD2DInputs[0], statePix(id, 0), 0);
		vec4 quat = texelFetch(sTD2DInputs[0], statePix(id, 1), 0);
		vec4 lin = texelFetch(sTD2DInputs[0], statePix(id, 2), 0);
		vec4 ang = texelFetch(sTD2DInputs[0], statePix(id, 3), 0);
		vec4 ext = texelFetch(sTD2DInputs[1], statePix(id, 0), 0);
		vec4 mid = texelFetch(sTD2DInputs[1], statePix(id, 1), 0);
		float sd = sdOriented(p, pose.xyz, quat, ext.xyz, mid.xyz);
		if (sd < best)
		{
			best = sd;
			vel = lin.xyz + cross(ang.xyz, p - pose.xyz);
		}
	}
	TDImageStoreOutput(0, c, vec4(best, vel));
}
