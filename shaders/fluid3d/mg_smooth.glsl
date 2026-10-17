#include <bus3d_glsl>
// Weighted Jacobi smoother (omega 2/3). One iteration per pass; pass k reads pass k-1 through input 0.
// Red-black Gauss-Seidel raced the previous-pass buffer on this GPU and diverged. Weighted Jacobi matches
// the stencil the projection uses and stays contractive. in0 pressure, in1 rhs, in2 obstacles.
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

bool solidAt(ivec3 nb, ivec3 res)
{
	vec3 uv = clamp((vec3(nb) + 0.5) / vec3(res), vec3(0.0), vec3(1.0));
	return texture(sTD3DInputs[2], uv).r < 0.0;
}

float pAt(ivec3 nb, ivec3 res, float pc)
{
	if (any(lessThan(nb, ivec3(0))) || any(greaterThanEqual(nb, res)))
		return pc;
	if (solidAt(nb, res))
		return pc;
	return texelFetch(sTD3DInputs[0], nb, 0).r;
}

void main()
{
	ivec3 c = busCoord();
	ivec3 res = ivec3(busRes());
	if (!busLive(c))
		return;
	float prev = texelFetch(sTD3DInputs[0], c, 0).r;
	if (solidAt(c, res))
	{
		TDImageStoreOutput(0, c, vec4(0.0));
		return;
	}
	// Red-black Gauss-Seidel: even passes update even cells, odd passes update odd cells.
	// The other color keeps the previous pass, which already holds the freshly updated neighbours.
	bool mine = ((c.x + c.y + c.z) & 1) == (uTDPass & 1);
	if (!mine)
	{
		TDImageStoreOutput(0, c, vec4(prev, 0.0, 0.0, 1.0));
		return;
	}
	float sum = pAt(c + ivec3(1, 0, 0), res, prev) + pAt(c - ivec3(1, 0, 0), res, prev)
	          + pAt(c + ivec3(0, 1, 0), res, prev) + pAt(c - ivec3(0, 1, 0), res, prev)
	          + pAt(c + ivec3(0, 0, 1), res, prev) + pAt(c - ivec3(0, 0, 1), res, prev);
	float h = busCell();
	float rhs = texelFetch(sTD3DInputs[1], c, 0).r;
	TDImageStoreOutput(0, c, vec4((sum - h * h * rhs) / 6.0, 0.0, 0.0, 1.0));
}
