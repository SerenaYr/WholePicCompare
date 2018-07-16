#pragma once
#include <opencv.hpp>


using namespace cv;
using namespace std;
/// <summary>
///ͼ����̬��
/// </summary>
class DealImageUtil
{
public:

	/// <summary>
	/// ͼ�������
	/// </summary>
	/// <param name="src">ͼ</param>
	/// <param name="size_w">�ߴ��С</param>
	/// <returns>cv.Mat.</returns>
	static Mat ImgClose(const Mat& src, int size_w);

	/// <summary>
	/// ������
	/// </summary>
	/// <param name="src">ͼ</param>
	/// <param name="size_w">�ߴ��С</param>
	/// <returns>cv.Mat.</returns>
	static Mat ImgOpen(const Mat& src, int size_w);

	/// <summary>
	/// ��ȡ�������
	/// </summary>
	/// <param name="src">��ֵͼ</param>
	/// <returns>��������</returns>
	static vector<Point> getMaxCountour(const Mat& src);

	/// <summary>
	///��ȡ����
	/// </summary>
	/// <param name="src">ͼ ��ֵͼ</param>
	/// <returns>�Ӵ�С������ͼ</returns>
	static vector<vector<Point>> getCountours(const Mat& src);


	/// <summary>
	/// ��˹ģ��
	/// </summary>
	/// <param name="src">����ͼ</param>
	/// <param name="dst">���</param>
	/// <param name="k">�˳߶ȴ�С</param>
	static void GaussianBlur(const Mat& src, Mat& dst, float k);



	/// <summary>
	/// ���ݸ�ʽת��
	/// </summary>
	/// <param name="src">The source.</param>
	/// <param name="dst">The DST.</param>
	template <class SrcType, class DstType>
	static void convertVec(std::vector<SrcType>& src, std::vector<DstType>& dst) {
		dst.resize(src.size());
		std::copy(src.begin(), src.end(), dst.begin());
	}

	/// <summary>
	/// ˮƽͶӰ
	/// </summary>
	/// <param name="src"></param>
	/// <returns></returns>
	static std::vector<double> horizontalProjection(const cv::Mat& src);

	/// <summary>
	/// ��ֱͶӰ
	/// </summary>
	/// <param name="src"></param>
	/// <returns></returns>
	static std::vector<double> verticalProjection(const cv::Mat& src);

	//�߽����
	template <typename T>
	static vector<int> findIndex(const vector<T>& vecT, double val)
	{
		vector<int> out;
		if (vecT.size() == 0) return out;
		vector<int> b_index;
		b_index.reserve(vecT.size());
		for (auto v : vecT)
		{
			b_index.push_back(v > val);
		}

		for (size_t i = 0, size = b_index.size() - 1; i < size; i++)
		{
			if (i == 0 && b_index[i])//��һ��
			{
				out.push_back(i);
				continue;
			}
			if (i == size - 1 && b_index[i])//���һ��
			{
				out.push_back(i);
				continue;
			}
			if (b_index[i] != b_index[i + 1])
			{
				out.push_back(i);
			}

		}
		return out;
	}


	/// <summary>
	/// ˮƽ����
	/// </summary>
	/// <param name="src">The source.</param>
	/// <param name="out">The out.</param>
	/// <param name="hor_thresh">The hor_thresh.</param>
	static void RLSA_H(const Mat& src, Mat& out, int hor_thresh);

	/// <summary>
	/// ��ֱ����
	/// </summary>
	/// <param name="src">The source.</param>
	/// <param name="out">The out.</param>
	/// <param name="hor_thresh">The hor_thresh.</param>
	static void RLSA_V(const Mat& src, Mat& out, int hor_thresh);
};

typedef  DealImageUtil DUtil;
