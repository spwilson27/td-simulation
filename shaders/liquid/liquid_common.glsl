// Shared FLIP helpers. Quadratic B-spline weights are a partition of unity for an interior
// particle (support 1.5 cells). Cell ids are exact in float32 up to 128^3.

float lqKernel(float d)
{
	float a = abs(d);
	if (a < 0.5)
		return 0.75 - d * d;
	if (a < 1.5)
		return 0.5 * (1.5 - a) * (1.5 - a);
	return 0.0;
}

float lqWeight(vec3 d)
{
	return lqKernel(d.x) * lqKernel(d.y) * lqKernel(d.z);
}

int lqLinear(ivec3 c, ivec3 res)
{
	return c.x + res.x * (c.y + res.y * c.z);
}

ivec2 lqTexel(int index, int width)
{
	int x = index - (index / width) * width;
	return ivec2(x, index / width);
}

// Histogram of a radix digit. Tiles stay in order. The texture is at most 4096 wide, so a
// 2M-particle sort (32768 tiles) wraps onto extra rows instead of exceeding the 16384 limit.
ivec2 lqHistCoord(int group, int digit, int tiles, int stride)
{
	int span = max(stride, 1);
	int rows = max((tiles + span - 1) / span, 1);
	int x = group - (group / span) * span;
	int y = digit * rows + group / span;
	return ivec2(x, y);
}

// 0 air, 1 liquid, 2 solid. Stored in the red channel.
#define LQ_AIR 0
#define LQ_LIQUID 1
#define LQ_SOLID 2

int lqKind(float stored)
{
	return int(stored + 0.5);
}
