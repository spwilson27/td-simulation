#include <bus3d_glsl>
// Divergence whose adjoint is the 7-point Laplacian the pressure solver inverts.
// A centred difference over two cells composes with the centred gradient to a wider stencil, so a
// correct-looking Jacobi step barely changes the divergence that shader measures. Each fluid cell
// owns the face on its +side: that flux is this cell's velocity when the neighbour is fluid, the
// obstacle velocity when either side is solid, and 0 outside the domain (a wall).
// in0 velocity, in1 obstacles, in2 scalars (a = expansion the projection should leave in).
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

bool inRange(ivec3 c, ivec3 res)
{
	return all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, res));
}

bool solidAt(ivec3 c)
{
	return busSolid(texelFetch(sTD3DInputs[1], c, 0));
}

float fluxX(ivec3 owner, ivec3 res)
{
	if (!inRange(owner, res))
		return 0.0;
	ivec3 nb = owner + ivec3(1, 0, 0);
	if (!inRange(nb, res))
	{
		// Open side: the fluid may leave, so this cell's velocity is the flux. A wall flux is 0.
		if (!busSideOpen(ivec3(1, 0, 0)))
			return 0.0;
		if (solidAt(owner))
			return busObstacleVel(texelFetch(sTD3DInputs[1], owner, 0)).x;
		return texelFetch(sTD3DInputs[0], owner, 0).x;
	}
	if (solidAt(owner))
		return busObstacleVel(texelFetch(sTD3DInputs[1], owner, 0)).x;
	if (solidAt(nb))
		return busObstacleVel(texelFetch(sTD3DInputs[1], nb, 0)).x;
	return texelFetch(sTD3DInputs[0], owner, 0).x;
}

float fluxY(ivec3 owner, ivec3 res)
{
	if (!inRange(owner, res))
		return 0.0;
	ivec3 nb = owner + ivec3(0, 1, 0);
	if (!inRange(nb, res))
	{
		if (!busSideOpen(ivec3(0, 1, 0)))
			return 0.0;
		if (solidAt(owner))
			return busObstacleVel(texelFetch(sTD3DInputs[1], owner, 0)).y;
		return texelFetch(sTD3DInputs[0], owner, 0).y;
	}
	if (solidAt(owner))
		return busObstacleVel(texelFetch(sTD3DInputs[1], owner, 0)).y;
	if (solidAt(nb))
		return busObstacleVel(texelFetch(sTD3DInputs[1], nb, 0)).y;
	return texelFetch(sTD3DInputs[0], owner, 0).y;
}

float fluxZ(ivec3 owner, ivec3 res)
{
	if (!inRange(owner, res))
		return 0.0;
	ivec3 nb = owner + ivec3(0, 0, 1);
	if (!inRange(nb, res))
	{
		if (!busSideOpen(ivec3(0, 0, 1)))
			return 0.0;
		if (solidAt(owner))
			return busObstacleVel(texelFetch(sTD3DInputs[1], owner, 0)).z;
		return texelFetch(sTD3DInputs[0], owner, 0).z;
	}
	if (solidAt(owner))
		return busObstacleVel(texelFetch(sTD3DInputs[1], owner, 0)).z;
	if (solidAt(nb))
		return busObstacleVel(texelFetch(sTD3DInputs[1], nb, 0)).z;
	return texelFetch(sTD3DInputs[0], owner, 0).z;
}

void main()
{
	ivec3 c = busCoord();
	ivec3 res = ivec3(busRes());
	if (!busLive(c))
		return;
	if (solidAt(c))
	{
		TDImageStoreOutput(0, c, vec4(0.0));
		return;
	}
	float div = fluxX(c, res) - fluxX(c - ivec3(1, 0, 0), res)
	          + fluxY(c, res) - fluxY(c - ivec3(0, 1, 0), res)
	          + fluxZ(c, res) - fluxZ(c - ivec3(0, 0, 1), res);
	div = div / busCell() - texelFetch(sTD3DInputs[2], c, 0).a;
	TDImageStoreOutput(0, c, vec4(div, 0.0, 0.0, 1.0));
}
