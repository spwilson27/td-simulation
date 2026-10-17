#include <bus3d_glsl>
// Weighted Jacobi smoother, omega 2/3. One iteration per pass; pass k reads pass k-1 through input 0.
// Red-black Gauss-Seidel on a read-write multipass raced the previous image on this GPU. Weighted Jacobi
// only reads that previous image, and it is the smoother for the same 7-point stencil as the projection.
// Obstacles are sampled in UV so a coarse level (this TOP's resolution) tests the solid at its own cell centre.
// in0 pressure, in1 rhs, in2 obstacles.
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

bool solidAt(ivec3 nb, ivec3 res)
{
	if (any(lessThan(nb, ivec3(0))) || any(greaterThanEqual(nb, res)))
		return true;
	vec3 uv = (vec3(nb) + 0.5) / vec3(res);
	return texture(sTD3DInputs[2], clamp(uv, vec3(0.0), vec3(1.0))).r < 0.0;
}

float pAt(ivec3 nb, ivec3 res, float pc, ivec3 dir)
{
	if (any(lessThan(nb, ivec3(0))) || any(greaterThanEqual(nb, res)))
		return busPressureGhost(dir, pc);
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
	float sum = pAt(c + ivec3(1, 0, 0), res, prev, ivec3(1, 0, 0)) + pAt(c - ivec3(1, 0, 0), res, prev, ivec3(-1, 0, 0))
	          + pAt(c + ivec3(0, 1, 0), res, prev, ivec3(0, 1, 0)) + pAt(c - ivec3(0, 1, 0), res, prev, ivec3(0, -1, 0))
	          + pAt(c + ivec3(0, 0, 1), res, prev, ivec3(0, 0, 1)) + pAt(c - ivec3(0, 0, 1), res, prev, ivec3(0, 0, -1));
	float h = busCell();
	float rhs = texelFetch(sTD3DInputs[1], c, 0).r;
	// omega 2/3. One pass reads only the previous pass, so it does not race the image it writes.
	const float omega = 2.0 / 3.0;
	float guess = (sum - h * h * rhs) / 6.0;
	TDImageStoreOutput(0, c, vec4((1.0 - omega) * prev + omega * guess, 0.0, 0.0, 1.0));
}
