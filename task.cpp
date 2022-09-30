#include "task.h"
#include <x86intrin.h>

bool isLineIntersect(
	const Point& p11, const Point& p12, const Point& p21, const Point& p22
)
{
	float a2x_sub_a1x = p12.x - p11.x;
	float a2y_sub_a1y = p12.y - p11.y;
	float b2x_sub_b1x = p22.x - p21.x;
	float b2y_sub_b1y = p22.y - p21.y;
	float a1y_sub_b1y = p11.y - p21.y;
	float a1x_sub_b1x = p11.x - p21.x;

	float det = (b2y_sub_b1y) * (a2x_sub_a1x)-(b2x_sub_b1x) * (a2y_sub_a1y);
	if (det == 0) return false;

	float a_factor = ((b2x_sub_b1x) * (a1y_sub_b1y)-(b2y_sub_b1y) * (a1x_sub_b1x)) / det;
	float b_factor = ((a2x_sub_a1x) * (a1y_sub_b1y)-(a2y_sub_a1y) * (a1x_sub_b1x)) / det;
	if (a_factor > 0.0f && a_factor < 1.0f && b_factor > 0.0f && b_factor < 1.0f) return true;

	return false;
}

int isLineIntersectSIMD(
	const Point& p111, const Point& p112, const Point& p121, const Point& p122,
	const Point& p211, const Point& p212, const Point& p221, const Point& p222,
	const Point& p311, const Point& p312, const Point& p321, const Point& p322,
	const Point& p411, const Point& p412, const Point& p421, const Point& p422,
	const Point& p511, const Point& p512, const Point& p521, const Point& p522,
	const Point& p611, const Point& p612, const Point& p621, const Point& p622,
	const Point& p711, const Point& p712, const Point& p721, const Point& p722,
	const Point& p811, const Point& p812, const Point& p821, const Point& p822
)
{
	static __m256 _zero = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0);
	static __m256 _one = _mm256_set_ps(1, 1, 1, 1, 1, 1, 1, 1);

	__m256 _a2x_sub_a1x, _a2x, _a1x;
	_a1x = _mm256_set_ps(p111.x, p211.x, p311.x, p411.x, p511.x, p611.x, p711.x, p811.x);
	_a2x = _mm256_set_ps(p112.x, p212.x, p312.x, p412.x, p512.x, p612.x, p712.x, p812.x);
	_a2x_sub_a1x = _mm256_sub_ps(_a2x, _a1x);

	__m256 _a2y_sub_a1y, _a2y, _a1y;
	_a1y = _mm256_set_ps(p111.y, p211.y, p311.y, p411.y, p511.y, p611.y, p711.y, p811.y);
	_a2y = _mm256_set_ps(p112.y, p212.y, p312.y, p412.y, p512.y, p612.y, p712.y, p812.y);
	_a2y_sub_a1y = _mm256_sub_ps(_a2y, _a1y);

	__m256 _b2x_sub_b1x, _b2x, _b1x;
	_b1x = _mm256_set_ps(p121.x, p221.x, p321.x, p421.x, p521.x, p621.x, p721.x, p821.x);
	_b2x = _mm256_set_ps(p122.x, p222.x, p322.x, p422.x, p522.x, p622.x, p722.x, p822.x);
	_b2x_sub_b1x = _mm256_sub_ps(_b2x, _b1x);

	__m256 _b2y_sub_b1y, _b2y, _b1y;
	_b1y = _mm256_set_ps(p121.y, p221.y, p321.y, p421.y, p521.y, p621.y, p721.y, p821.y);
	_b2y = _mm256_set_ps(p122.y, p222.y, p322.y, p422.y, p522.y, p622.y, p722.y, p822.y);
	_b2y_sub_b1y = _mm256_sub_ps(_b2y, _b1y);

	__m256 _a1y_sub_b1y, _a1x_sub_b1x;
	_a1y_sub_b1y = _mm256_sub_ps(_a1y, _b1y);
	_a1x_sub_b1x = _mm256_sub_ps(_a1x, _b1x);

	__m256 _det, _a_factor, _b_factor;
	_det = _mm256_fmsub_ps(_b2y_sub_b1y, _a2x_sub_a1x, _mm256_mul_ps(_b2x_sub_b1x, _a2y_sub_a1y));
	_a_factor = _mm256_div_ps(_mm256_fmsub_ps(_b2x_sub_b1x, _a1y_sub_b1y, _mm256_mul_ps(_b2y_sub_b1y, _a1x_sub_b1x)), _det);
	_b_factor = _mm256_div_ps(_mm256_fmsub_ps(_a2x_sub_a1x, _a1y_sub_b1y, _mm256_mul_ps(_a2y_sub_a1y, _a1x_sub_b1x)), _det);

	__m256 _a_factor_gt_zero_lt_one, _b_factor_gt_zero_lt_one;
	_a_factor_gt_zero_lt_one = _mm256_and_ps(_mm256_cmp_ps(_a_factor, _zero, _CMP_GT_OS), _mm256_cmp_ps(_a_factor, _one, _CMP_LT_OS));
	_b_factor_gt_zero_lt_one = _mm256_and_ps(_mm256_cmp_ps(_b_factor, _zero, _CMP_GT_OS), _mm256_cmp_ps(_b_factor, _one, _CMP_LT_OS));

	int _a_and_b_factor_mask = _mm256_movemask_ps(_mm256_and_ps(_a_factor_gt_zero_lt_one, _b_factor_gt_zero_lt_one));

	if (_a_and_b_factor_mask != 0) return true;

	return false;
}

bool isTriangleIntersect(const Triangle& tri1, const Triangle& tri2)
{
	if (isLineIntersectSIMD(
		tri1.a, tri1.b, tri2.a, tri2.b,
		tri1.a, tri1.b, tri2.b, tri2.c,
		tri1.a, tri1.b, tri2.c, tri2.a,
		tri1.b, tri1.c, tri2.a, tri2.b,
		tri1.b, tri1.c, tri2.b, tri2.c,
		tri1.b, tri1.c, tri2.c, tri2.a,
		tri1.c, tri1.a, tri2.a, tri2.b,
		tri1.c, tri1.a, tri2.b, tri2.c
	))
	{
		return true;
	}
	return isLineIntersect(tri1.c, tri1.a, tri2.c, tri2.a);
}

void Task::checkIntersections(const std::vector<Triangle>& in_triangles, std::vector<int>& out_count)
{
	if (!(1 <= in_triangles.size() && in_triangles.size() <= 100000)) return;

	out_count.resize(in_triangles.size());

	for (unsigned i = 0; i < in_triangles.size(); i++)
	{
		for (unsigned l = i + 1; l < in_triangles.size(); l++)
		{
			if (isTriangleIntersect(in_triangles[i], in_triangles[l]))
			{
				out_count[i] += 1;
				out_count[l] += 1;
			}
		}
	}
}
