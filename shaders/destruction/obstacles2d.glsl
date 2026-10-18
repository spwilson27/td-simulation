#include <bus_glsl>
// 2D slice of the same oriented boxes. in0 state, in1 extents, in2 bins. uCount matches the 3D publisher.
// uSlice.x is the world z of this slice; a box contributes when the slice cuts it.
uniform vec4 uCount;
uniform vec4 uSlice;

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

out vec4 fragColor;
void main()
{
	vec2 p2 = busUVToWorld(vUV.st);
	vec3 p = vec3(p2, uSlice.x);
	vec3 bmin = vec3(uBounds.x, uBounds.y, uSlice.y);
	vec3 bmax = vec3(uBounds.z, uBounds.w, uSlice.z);
	vec3 size = max(bmax - bmin, vec3(1e-4));
	int grid = max(int(uCount.y + 0.5), 1);
	ivec3 bin = ivec3(clamp((p - bmin) / size * float(grid), vec3(0.0), vec3(float(grid) - 1.0)));
	int stride = max(int(uCount.z + 0.5), 1);
	int start = (bin.x + grid * (bin.y + grid * bin.z)) * stride;
	int count = int(texelFetch(sTD2DInputs[2], binPix(start), 0).r + 0.5);
	float best = BUS_EMPTY_DISTANCE;
	vec2 vel = vec2(0.0);
	for (int i = 0; i < 8; i++)
	{
		if (i >= count)
			break;
		int texel = i / 4;
		vec4 packed = texelFetch(sTD2DInputs[2], binPix(start + 1 + texel), 0);
		int id = int(packed[i - texel * 4] + 0.5);
		if (id < 0 || id >= int(uCount.x + 0.5))
			continue;
		vec4 pose = texelFetch(sTD2DInputs[0], statePix(id, 0), 0);
		vec4 quat = texelFetch(sTD2DInputs[0], statePix(id, 1), 0);
		vec4 lin = texelFetch(sTD2DInputs[0], statePix(id, 2), 0);
		vec4 ang = texelFetch(sTD2DInputs[0], statePix(id, 3), 0);
		vec4 ext = texelFetch(sTD2DInputs[1], statePix(id, 0), 0);
		vec4 mid = texelFetch(sTD2DInputs[1], statePix(id, 1), 0);
		vec3 local = qrot(vec4(-quat.xyz, quat.w), p - pose.xyz) - mid.xyz;
		vec3 q = abs(local) - max(ext.xyz, vec3(1e-4));
		float sd = length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
		if (sd < best)
		{
			best = sd;
			vec3 wvel = lin.xyz + cross(ang.xyz, p - pose.xyz);
			vel = wvel.xy;
		}
	}
	fragColor = TDOutputSwizzle(vec4(best, vel, 0.0));
}
