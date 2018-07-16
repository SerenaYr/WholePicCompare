#include <iostream>
#include <opencv.hpp>
#include "DealImageUtil.h"
#include <ctime>
#include<fstream>

using namespace cv;
using namespace std;

void myshow(char* name, Mat img) {
	//	cv::namedWindow(name, 0);
	namedWindow(name, CV_WINDOW_KEEPRATIO);
	imshow(name, img);
}
void myshow(Mat img, char* name) {
	//	cv::namedWindow(name, 0);
	namedWindow(name, CV_WINDOW_KEEPRATIO);
	imshow(name, img);
}

//ģ�ͼ�⣺�ֱ��ͼ������������������м��
void DealModel(const Mat &src, vector<Rect> &targ_search, vector<Rect> &targ_up_TuiJian, Rect &targ_up,
	vector<Rect> &rects_middle_targ, vector<Rect> &rects_middle_targ_text,
	vector<Rect> &rects_down_targ, vector<Rect> &rects_down_targ_text, vector<Rect> &rects_down_targ_two)
{
	Mat mImg = src;

	//����ƥ��ͼƬ�ֱ𻮷�Ϊrect_up��rect_middle��rect_down�������򣬷ֱ���Rect���ͱ�������
	//0,450|1079,780 ��ͼ1�Ĵ�������ͨ�����ϽǼ����½�����ȷ�����Σ�
	//0,840|1079,1010
	//0,1240|1079,2060

	/*cout << mImg.cols << endl;
	cout << mImg.rows << endl;*/

	const Rect rect_up_Search(Point(0, 110), Point(mImg.cols, 250));	//���ϱ�������
	const Rect rect_up_TuiJian(Point(100, 290), Point(700, 392));		//���ϱ����֡��ֿ⡱��
	const Rect rect_up(Point(0, 450), Point(mImg.cols, 780));			//���ϱߴ�ͼ
	const Rect rect_middle(Point(0, 840), Point(mImg.cols, 1010));		//�м�ͼ������
	const Rect rect_middle_BiTing(Point(0, 1110), Point(mImg.cols, 1190));//�м����֡������赥��
																		  //const Rect rect_down(Point(0, 1240), Point(mImg.cols, 2070));		//�±�6��ר��ͼƬ
	const Rect rect_down(Point(0, 1240), Point(mImg.cols, 1540));		//�±�3��ר��ͼƬ
	const Rect rect_botton_GeQuName(Point(0, 1570), Point(mImg.cols, 1700));//�����һ��3��ͼƬ�ĸ�������
	const Rect rect_down_two(Point(0, 1760), Point(mImg.cols, 2070));	//���±�3��ר��ͼƬ

	Mat cannMat;
	//cv::Canny(mImg, cannMat, 50, 150);
	cv::Canny(mImg, cannMat, 10, 50);					//������Ĵ����ͼƬ����Ե���

														//--------------------------------------- ���Ϸ���������λ ---------------------------------------

	Mat tempUpSearch;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rect_up_Search), tempUpSearch, 5);			// ����ˮƽ���ؾ���С��35�ĵ�
	DUtil::RLSA_V(tempUpSearch, tempUpSearch, 5);						// ���Ӵ�ֱ���ؾ���С��35�ĵ�
																		// �����ͨͼ�������������Ӧ����4�������б�Ե��λ����4�������ϣ�
	vector<vector<Point>> countours_up_search = DUtil::getCountours(tempUpSearch);

	for (auto countour_up_search : countours_up_search)
	{
		// ������������
		Rect rect_up_search = cv::boundingRect(countour_up_search);
		rect_up_search = rect_up_search + rect_up_Search.tl();			// rect_middle.tl����ԭ�м���������ϽǶ�������
		if (rect_up_search.x > 170 && rect_up_search.x < 900)
			continue;
		targ_search.push_back(rect_up_search);		// rects_middle_targ�������������׼ȷ�ľ���������Ϣ������Ӧ�ð���4��Rect��Ϣ��

													// ��ʾ������ʱ���źž�������
		cv::rectangle(mImg, rect_up_search, Scalar(0, 255, 0), 8);
	}

	// ����������ʼ��xλ�ö�Rect��������������ս��Ϊrects_middle_targ�����δ洢�м�����ҵ�4��Rect��Ϣ��
	sort(targ_search.begin(), targ_search.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- ���Ϸ����ֶ�λ ---------------------------------------

	Mat tempUpText;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rect_up_TuiJian), tempUpText, 10);			// ����ˮƽ���ؾ���С��35�ĵ�
	DUtil::RLSA_V(tempUpText, tempUpText, 10);						// ���Ӵ�ֱ���ؾ���С��35�ĵ�
																	// �����ͨͼ�������������Ӧ����4�������б�Ե��λ����4�������ϣ�
	vector<vector<Point>> countours_up_text = DUtil::getCountours(tempUpText);

	for (auto countour_up_text : countours_up_text)
	{
		// ������������
		Rect rect_up_text = cv::boundingRect(countour_up_text);
		rect_up_text = rect_up_text + rect_up_TuiJian.tl();			// rect_middle.tl����ԭ�м���������ϽǶ�������
		targ_up_TuiJian.push_back(rect_up_text);		// rects_middle_targ�������������׼ȷ�ľ���������Ϣ������Ӧ�ð���4��Rect��Ϣ��

														// ��ʾ���������־�������
		cv::rectangle(mImg, rect_up_text, Scalar(0, 255, 0), 8);
	}

	// ����������ʼ��xλ�ö�Rect��������������ս��Ϊrects_middle_targ�����δ洢�м�����ҵ�4��Rect��Ϣ��
	sort(targ_up_TuiJian.begin(), targ_up_TuiJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- ���Ϸ���ͼ��λ ---------------------------------------

	//ˮƽ��ֱͶӰ
	vector<double>  hproj = DUtil::horizontalProjection(cannMat(rect_up));	//ˮƽͶӰ����
	vector<double>  vproj = DUtil::verticalProjection(cannMat(rect_up));	//��ֱͶӰ����

																			//���ұ߽�
	vector<int> hds = DUtil::findIndex(hproj, 15);	//�����г���15�����ص㿪ʼ��Ϊͼ������
	vector<int> vds = DUtil::findIndex(vproj, 15);	//�����г���15�����ص㿪ʼ��Ϊͼ������

	Point pt1 = Point(hds[0], vds[0]);
	Point pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);
	targ_up = Rect(pt1, pt2);						//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ

													//ƫ�ƣ���ͼ�����ԭ����Rect���Ծ���λ�ý���λ�ƣ�����targ_up�д洢׼ȷʶ������Ϸ���ͼ��Rect��Ϣ
	targ_up.x += rect_up.x;
	targ_up.y += rect_up.y;

	//--------------------------------------- �м�4��Сͼ����λ ---------------------------------------

	Mat tempMiddle;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rect_middle), tempMiddle, 35);			// ����ˮƽ���ؾ���С��35�ĵ�
	DUtil::RLSA_V(tempMiddle, tempMiddle, 35);						// ���Ӵ�ֱ���ؾ���С��35�ĵ�

																	// �����ͨͼ�������������Ӧ����4�������б�Ե��λ����4�������ϣ�
	vector<vector<Point>> countours = DUtil::getCountours(tempMiddle);
	//	vector<Rect> rects_middle_targ;

	for (auto countour : countours)
	{
		//������������
		Rect rect = cv::boundingRect(countour);
		rect = rect + rect_middle.tl();			// rect_middle.tl����ԭ�м���������ϽǶ�������
		rects_middle_targ.push_back(rect);		// rects_middle_targ�������������׼ȷ�ľ���������Ϣ������Ӧ�ð���4��Rect��Ϣ��
	}

	// ����������ʼ��xλ�ö�Rect��������������ս��Ϊrects_middle_targ�����δ洢�м�����ҵ�4��Rect��Ϣ��
	sort(rects_middle_targ.begin(), rects_middle_targ.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- �м䡰�����赥�����ֶ�λ ---------------------------------------

	Mat tempMiddleText;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rect_middle_BiTing), tempMiddleText, 40);			// ����ˮƽ���ؾ���С��35�ĵ�
	DUtil::RLSA_V(tempMiddleText, tempMiddleText, 40);						// ���Ӵ�ֱ���ؾ���С��35�ĵ�
																			// �����ͨͼ�������������Ӧ����4�������б�Ե��λ����4�������ϣ�
	vector<vector<Point>> countours_middle_text = DUtil::getCountours(tempMiddleText);

	for (auto countour_middle_text : countours_middle_text)
	{
		// ������������
		Rect rect_middle_text = cv::boundingRect(countour_middle_text);
		rect_middle_text = rect_middle_text + rect_middle_BiTing.tl();			// rect_middle.tl����ԭ�м���������ϽǶ�������
		rects_middle_targ_text.push_back(rect_middle_text);		// rects_middle_targ�������������׼ȷ�ľ���������Ϣ������Ӧ�ð���4��Rect��Ϣ��

																// ��ʾ�м����־�������
		cv::rectangle(mImg, rect_middle_text, Scalar(0, 255, 0), 8);
	}

	// ����������ʼ��xλ�ö�Rect��������������ս��Ϊrects_middle_targ�����δ洢�м�����ҵ�4��Rect��Ϣ��
	sort(rects_middle_targ_text.begin(), rects_middle_targ_text.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- ����3��ͼƬ����λ ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rect_down));
	vproj = DUtil::verticalProjection(cannMat(rect_down));

	//���ұ߽�
	double th = rect_down.width*0.1;
	vds = DUtil::findIndex(vproj, th);		// �����ص㳬�����*0.1�ı߽��б�ǣ�����׼ȷ���η�Χ
	hds = DUtil::findIndex(hproj, th);
	pt1 = Point(hds[0], vds[0]);
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);
	Rect rect_down_targ = Rect(pt1, pt2);

	//ƫ��
	rect_down_targ.y += rect_down.y;

	//ˮƽ�ʹ�ֱͶӰ
	DUtil::RLSA_H(cannMat(rect_down_targ), tempMiddle, 5);	// �����ؾ���С��5�����ص��������
	DUtil::RLSA_V(tempMiddle, tempMiddle, 5);

	//�������ң�Ӧ����6��������
	countours = DUtil::getCountours(tempMiddle);
	//	vector<Rect> &rects_down_targ

	vector<Rect> tempRect1;								// �ֱ𱣴�6������ͼ���Rect��Ϣ��ÿ��vector�б���3��ͼƬ��Ϣ��

	for (int i = 0; i < 3; i++)
	{
		//������������
		Rect rect = cv::boundingRect(countours[i]);
		rect = rect + rect_down_targ.tl();			// ���ݾ��δ��������ϽǶ����������ƽ��
		tempRect1.push_back(rect);				// ����һ��ͼ����Ϣ�洢��tempRect1�����У�
	}

	//����������
	sort(tempRect1.begin(), tempRect1.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//��¼���ݣ����մ��ϵ��£������ҵ�˳��6��ͼƬ��Ϣ���δ洢��rects_down_targ�����У�
	for (int i = 0; i < tempRect1.size(); i++)
	{
		rects_down_targ.push_back(tempRect1[i]);
	}

	//--------------------------------------- �����һ��3��ͼƬ���ֶ�λ ---------------------------------------

	Mat tempBottonText;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rect_botton_GeQuName), tempBottonText, 30);			// ����ˮƽ���ؾ���С��35�ĵ�
	DUtil::RLSA_V(tempBottonText, tempBottonText, 30);						// ���Ӵ�ֱ���ؾ���С��35�ĵ�
																			// �����ͨͼ�������������Ӧ����4�������б�Ե��λ����4�������ϣ�
	vector<vector<Point>> countours_botton_text = DUtil::getCountours(tempBottonText);

	for (auto countour_botton_text : countours_botton_text)
	{
		// ������������
		Rect rect_botton_text = cv::boundingRect(countour_botton_text);
		rect_botton_text = rect_botton_text + rect_botton_GeQuName.tl();			// rect_middle.tl����ԭ�м���������ϽǶ�������
		rects_down_targ_text.push_back(rect_botton_text);		// rects_middle_targ�������������׼ȷ�ľ���������Ϣ������Ӧ�ð���4��Rect��Ϣ��

																// ��ʾ�м����־�������
		cv::rectangle(mImg, rect_botton_text, Scalar(0, 255, 0), 8);
	}

	// ����������ʼ��xλ�ö�Rect��������������ս��Ϊrects_middle_targ�����δ洢�м�����ҵ�4��Rect��Ϣ��
	sort(rects_down_targ_text.begin(), rects_down_targ_text.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- ������3��ͼƬ����λ ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rect_down_two));
	vproj = DUtil::verticalProjection(cannMat(rect_down_two));

	//���ұ߽�
	double th2 = rect_down_two.width*0.1;
	vds = DUtil::findIndex(vproj, th2);		// �����ص㳬�����*0.1�ı߽��б�ǣ�����׼ȷ���η�Χ
	hds = DUtil::findIndex(hproj, th2);
	pt1 = Point(hds[0], vds[0]);
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);
	Rect rect_down_targ_two = Rect(pt1, pt2);

	//ƫ��
	rect_down_targ_two.y += rect_down_two.y;

	//ˮƽ�ʹ�ֱͶӰ
	DUtil::RLSA_H(cannMat(rect_down_targ_two), tempMiddle, 5);	// �����ؾ���С��5�����ص��������
	DUtil::RLSA_V(tempMiddle, tempMiddle, 5);

	//�������ң�Ӧ����6��������
	countours = DUtil::getCountours(tempMiddle);
	//	vector<Rect> &rects_down_targ

	vector<Rect> tempRect2;								// �ֱ𱣴�6������ͼ���Rect��Ϣ��ÿ��vector�б���3��ͼƬ��Ϣ��

	for (int i = 0; i < 3; i++)
	{
		//������������
		Rect rect2 = cv::boundingRect(countours[i]);
		rect2 = rect2 + rect_down_targ_two.tl();			// ���ݾ��δ��������ϽǶ����������ƽ��
		tempRect2.push_back(rect2);				// ����һ��ͼ����Ϣ�洢��tempRect1������
	}

	//����������
	sort(tempRect2.begin(), tempRect2.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//��¼���ݣ����մ��ϵ��£������ҵ�˳��6��ͼƬ��Ϣ���δ洢��rects_down_targ�����У�
	for (int i = 0; i < tempRect2.size(); i++)
	{
		rects_down_targ_two.push_back(tempRect2[i]);
	}
}

//���ͼ��λ
void DealDetectImg(const Mat &src, vector<Rect> &targ_up_search_detect, vector<Rect> &targ_up_text, Rect &targ_up,
	vector<Rect> &rects_middle, vector<Rect> &rects_middle_text,
	vector<Rect> &down_rects, vector<Rect> &down_rects_text, vector<Rect> &down_rects_two)
{
	Mat imgsrc = src;
	Mat thMat;

	Mat mImg = src;

	//��ɫ���ˣ�ȥ����ɫ������
	inRange(imgsrc, Scalar(240, 240, 240), Scalar(255, 255, 255), thMat);
	thMat = ~thMat;

	//����ͼƬ����ⷶΧ�������µ�����
	Point ptRightDown(imgsrc.cols - 1, 0.92*imgsrc.rows);	// ���������·����Ի��˵�ͼ����
	Rect targRect(Point(0, 0), ptRightDown);

	Mat cannMat;
	DUtil::GaussianBlur(imgsrc, imgsrc, 5);	//��˹ģ��
	cv::Canny(imgsrc, cannMat, 20, 30);		//��Ե���

	Mat cannMatClose = DUtil::ImgClose(cannMat, 10);	//������
	DUtil::RLSA_H(cannMat, cannMatClose, 10);			//��ˮƽ�ʹ�ֱ�������ؾ���С��10�����ص��������
	DUtil::RLSA_V(cannMatClose, cannMatClose, 10);

	// �Ӿ�ͼ����λ����Ϣ
	const Rect rect_up_Search(Point(0, 100), Point(imgsrc.cols, 0.1*imgsrc.rows));	//���ϱ�������
	const Rect rect_up_TuiJian(Point(0.064*imgsrc.cols, 0.13*imgsrc.rows), Point(0.67*imgsrc.cols, 0.19*imgsrc.rows));		//���ϱ����֡��ֿ⡱��
																															//const Rect rect_up(Point(0, 450), Point(mImg.cols, 780));			//���ϱߴ�ͼ
																															//const Rect rect_middle(Point(0, 840), Point(mImg.cols, 1010));		//�м�ͼ������
	const Rect rect_middle_BiTing(Point(0.055*imgsrc.cols, 0.58*imgsrc.rows), Point(0.95*imgsrc.cols, 0.62*imgsrc.rows));//�м����֡������赥��
																														 //																	  //const Rect rect_down(Point(0, 1240), Point(mImg.cols, 2070));		//�±�6��ר��ͼƬ
																														 //const Rect rect_down(Point(0, 1240), Point(mImg.cols, 1540));		//�±�3��ר��ͼƬ
	const Rect rect_botton_GeQuName(Point(51, 0.818*imgsrc.rows), Point(0.9*imgsrc.cols, 0.87*imgsrc.rows));//�����һ��3��ͼƬ�ĸ�������
	cv::rectangle(mImg, rect_botton_GeQuName, Scalar(0, 0, 255), 8);
	//const Rect rect_down_two(Point(0, 1760), Point(mImg.cols, 2070));	//���±�3��ר��ͼƬ

	//------------------------------------ ��ȡ������������λ����Ϣ ------------------------------------

	Mat tempUpSearch;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rect_up_Search), tempUpSearch, 5);			// ����ˮƽ���ؾ���С��35�ĵ�
	DUtil::RLSA_V(tempUpSearch, tempUpSearch, 5);						// ���Ӵ�ֱ���ؾ���С��35�ĵ�
																		// �����ͨͼ�������������Ӧ����4�������б�Ե��λ����4�������ϣ�
	vector<vector<Point>> countours_up_search = DUtil::getCountours(tempUpSearch);

	for (auto countour_up_search : countours_up_search)
	{
		// ������������
		Rect rect_up_search = cv::boundingRect(countour_up_search);
		rect_up_search = rect_up_search + rect_up_Search.tl();			// rect_middle.tl����ԭ�м���������ϽǶ�������
		if (rect_up_search.x > 170 && rect_up_search.x < 900)
			continue;
		targ_up_search_detect.push_back(rect_up_search);		// rects_middle_targ�������������׼ȷ�ľ���������Ϣ������Ӧ�ð���4��Rect��Ϣ��

		cv::rectangle(mImg, rect_up_search, Scalar(0, 255, 0), 8);	// ��ʾ������ʱ���źž�������
	}

	// ����������ʼ��xλ�ö�Rect��������������ս��Ϊrects_middle_targ�����δ洢�м�����ҵ�4��Rect��Ϣ��
	sort(targ_up_search_detect.begin(), targ_up_search_detect.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- ���Ϸ����Ƽ������ֶ�λ ---------------------------------------

	Mat tempUpText;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rect_up_TuiJian), tempUpText, 10);			// ����ˮƽ���ؾ���С��35�ĵ�
	DUtil::RLSA_V(tempUpText, tempUpText, 10);						// ���Ӵ�ֱ���ؾ���С��35�ĵ�
																	// �����ͨͼ�������������Ӧ����4�������б�Ե��λ����4�������ϣ�
	vector<vector<Point>> countours_up_text = DUtil::getCountours(tempUpText);

	for (auto countour_up_text : countours_up_text)
	{
		// ������������
		Rect rect_up_text = cv::boundingRect(countour_up_text);
		rect_up_text = rect_up_text + rect_up_TuiJian.tl();			// rect_middle.tl����ԭ�м���������ϽǶ�������
		targ_up_text.push_back(rect_up_text);		// rects_middle_targ�������������׼ȷ�ľ���������Ϣ������Ӧ�ð���4��Rect��Ϣ��

		cv::rectangle(mImg, rect_up_text, Scalar(0, 255, 0), 8);	// ��ʾ���������־�������
	}

	// ����������ʼ��xλ�ö�Rect��������������ս��Ϊrects_middle_targ�����δ洢�м�����ҵ�4��Rect��Ϣ��
	sort(targ_up_text.begin(), targ_up_text.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//------------------------------------ ��ȡ������ͼƬλ����Ϣ ------------------------------------
	vector<vector<Point>> countours = DUtil::getCountours(cannMat(targRect));
	targ_up = cv::boundingRect(countours[0]);	//�ҵ�������������Ĵ�ͼ

	cv::rectangle(mImg, targ_up, Scalar(0, 255, 0), 8);

	//------------------------------------ ��ȡ�м�4��Сͼ��λ����Ϣ ------------------------------------
	//�����м�����

	Rect rect_middle_temp = targ_up;	//�м�����
	rect_middle_temp.y = rect_middle_temp.br().y*1.05;		// y����ʼ���������ͼƬ�±�0.05λ�ÿ�ʼ���
	rect_middle_temp.height *= 0.8;							// �м����yֵ���Ϊ�������yֵ��0.8��

	Mat tempMiddle;

	//ˮƽ����
	DUtil::RLSA_H(cannMat(rect_middle_temp), tempMiddle, 35);
	DUtil::RLSA_V(tempMiddle, tempMiddle, 35);

	countours = DUtil::getCountours(tempMiddle);
	//	vector<Rect> rects_middle;

	for (auto countour : countours)
	{
		//������������
		Rect rectTemp = cv::boundingRect(countour);
		rectTemp = rectTemp + rect_middle_temp.tl();
		if (rectTemp.width < 100)
			continue;
		rects_middle.push_back(rectTemp);
		cv::rectangle(imgsrc, rectTemp, Scalar(255, 0, 255), 8);
	}
	//����������
	sort(rects_middle.begin(), rects_middle.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- �м䡰�����赥�����ֶ�λ ---------------------------------------

	Mat tempMiddleText;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rect_middle_BiTing), tempMiddleText, 40);			// ����ˮƽ���ؾ���С��35�ĵ�
	DUtil::RLSA_V(tempMiddleText, tempMiddleText, 40);						// ���Ӵ�ֱ���ؾ���С��35�ĵ�
																			// �����ͨͼ�������������Ӧ����4�������б�Ե��λ����4�������ϣ�
	vector<vector<Point>> countours_middle_text = DUtil::getCountours(tempMiddleText);

	for (auto countour_middle_text : countours_middle_text)
	{
		// ������������
		Rect rect_middle_text = cv::boundingRect(countour_middle_text);
		rect_middle_text = rect_middle_text + rect_middle_BiTing.tl();			// rect_middle.tl����ԭ�м���������ϽǶ�������
		rects_middle_text.push_back(rect_middle_text);		// rects_middle_targ�������������׼ȷ�ľ���������Ϣ������Ӧ�ð���4��Rect��Ϣ��

		cv::rectangle(mImg, rect_middle_text, Scalar(0, 255, 0), 8);	// ��ʾ�м����־�������
	}

	// ����������ʼ��xλ�ö�Rect��������������ս��Ϊrects_middle_targ�����δ洢�м�����ҵ�4��Rect��Ϣ��
	sort(rects_middle_text.begin(), rects_middle_text.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- �����һ��3��ͼƬ���ֶ�λ ---------------------------------------

	Mat tempBottonText;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rect_botton_GeQuName), tempBottonText, 30);			// ����ˮƽ���ؾ���С��35�ĵ�
	DUtil::RLSA_V(tempBottonText, tempBottonText, 30);						// ���Ӵ�ֱ���ؾ���С��35�ĵ�
																			// �����ͨͼ�������������Ӧ����4�������б�Ե��λ����4�������ϣ�
	vector<vector<Point>> countours_botton_text = DUtil::getCountours(tempBottonText);

	for (auto countour_botton_text : countours_botton_text)
	{
		// ������������
		Rect rect_botton_text = cv::boundingRect(countour_botton_text);
		rect_botton_text = rect_botton_text + rect_botton_GeQuName.tl();			// rect_middle.tl����ԭ�м���������ϽǶ�������
		down_rects_text.push_back(rect_botton_text);		// rects_middle_targ�������������׼ȷ�ľ���������Ϣ������Ӧ�ð���4��Rect��Ϣ��

		cv::rectangle(mImg, rect_botton_text, Scalar(0, 255, 0), 8);	// ��ʾ�м����־�������
	}

	// ����������ʼ��xλ�ö�Rect��������������ս��Ϊrects_middle_targ�����δ洢�м�����ҵ�4��Rect��Ϣ��
	sort(down_rects_text.begin(), down_rects_text.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//------------------------------------ ��ȡ����6��ͼƬλ����Ϣ ------------------------------------
	//Rect rect_down = Rect(Point(0, rect_middle_temp.y + rect_middle_temp.height*1.4),ptRightDown);
	Rect rect_down = Rect(Point(0, 0.65*imgsrc.rows), ptRightDown);
	Mat tempMat = thMat(rect_down) | cannMatClose(rect_down);

	countours = DUtil::getCountours(tempMat);
	float width_rect = 0;
	float height_rect = 0;
	int miny = 30000, maxy = 0;

	for (int j = 0; j < 3; j++)
	{
		Rect rect0 = cv::boundingRect(countours[j]);	//�������Ĵ�ͼ
		rect0.y += rect_down.y;
		width_rect += rect0.width;		//����ۼ�
		height_rect += rect0.height;
		down_rects.push_back(rect0);

		miny = min(miny, rect0.y);		//���ֵ
		maxy = max(maxy, rect0.y);
	}
	/********************************************/
	width_rect /= 3;
	height_rect /= 3;
	for (int j = 3; j < countours.size(); j++)
	{
		Rect rect0 = cv::boundingRect(countours[j]);//�������Ĵ�ͼ
		if (down_rects.size() == 6)//6�����˳�
			break;
		float rw = rect0.width / width_rect;
		float rh = rect0.height / height_rect;

		if (rw > 0.8 && rw < 1.3 && rh > 0.4 && rh < 1.2)
		{
			rect0.y += rect_down.y;
			down_rects.push_back(rect0);
			miny = min(miny, rect0.y);
			maxy = max(maxy, rect0.y);
			cv::rectangle(imgsrc, rect0, Scalar(255, 0, 0), 8);
		}
	}
	//3����ֻ��⵽һ��ͼƬ��
	if (down_rects.size() == 3)
	{
		sort(down_rects.begin(), down_rects.end(), [](const Rect &r1, const Rect &r2)	//ֻ������3��ͼƬλ�ý��м��
		{return r1.x < r2.x; });
	}
	else if (down_rects.size() == 6)
	{
		int thy = (miny + maxy) / 2;
		vector<Rect> tempRect1, tempRect2;
		for (int j = 0; j < 6; j++)
		{
			//������������
			Rect rect = down_rects[j];
			if (rect.y > thy)//���·���
			{
				tempRect1.push_back(rect);
			}
			else
				tempRect2.push_back(rect);
		}

		//�������������ŵڶ��ţ����ŵ�һ�ţ�
		sort(tempRect1.begin(), tempRect1.end(), [](const Rect &r1, const Rect &r2)
		{return r1.x < r2.x; });

		sort(tempRect2.begin(), tempRect2.end(), [](const Rect &r1, const Rect &r2)
		{return r1.x < r2.x; });

		down_rects.clear();

		//��¼����
		for (int i = 0; i < tempRect2.size(); i++)
		{
			down_rects.push_back(tempRect2[i]);
		}
		for (int i = 0; i < tempRect1.size(); i++)
		{
			down_rects.push_back(tempRect1[i]);
		}
	}
}

// ����ƥ���
float MatchRate(Rect r1, Rect r2)
{
	Rect rr0 = r1 & r2;			// ��������ȡ����
	Rect rr1 = r2 | r1;			// ��������ȡ����
	return  rr0.area() / (float)rr1.area();		// ����ƥ���=ͼ���ص����/�ܺ����
}


void main()
{
	//��ȡͼƬ
	Mat mImg = imread("1.jpg");				//���������ƥ��ͼ��
	Size msize = mImg.size();				//�����ƥ��ͼ���С����Ϊģ��
	Mat libImgSrc = imread("2.jpg");		//�����Ӿ���ͼƬ

											//�Ӿ�ͼ��ÿ��ͼƬ��ȼ��߶ȶ��壨��ģ����Ϊ���ݣ�
	int width = 450;
	int height = 799;

	//���Ӿ�ͼ�ֶ�����Ϊ5��ͼƬ���������ػ�������
	Rect rect1(0, 0, width, height);
	Rect rect2(450, 0, width, height);
	Rect rect3(900, 0, width, height);
	Rect rect4(1350, 0, width, height);
	Rect rect5(1800, 0, 674, 1199);

	//Rect��������rects�����α����Ӿ����е�5��ͼƬ
	vector<Rect> rects;
	rects.push_back(rect1);
	rects.push_back(rect2);
	rects.push_back(rect3);
	rects.push_back(rect4);
	rects.push_back(rect5);

	//ģ�ͼ�⣨Rect�����/����targ_up��rects_middle_targ��rects_down_targ�ֱ𱣴�ͼƬ��������������Ϣ��
	vector<Rect> targ_top_time;				// ���Ϸ�ʱ���ź���Ϣ��ʾ
	vector<Rect> targ_search;				// �Ϸ�������������λ��
	vector<Rect> targ_up_TuiJian;			// ������Ϸ����Ƽ��ֿ�����֡�
	Rect		 targ_up;					// ���Ϸ���ͼ
	vector<Rect> rects_middle_targ;			// �м�4��Сͼ
	vector<Rect> rects_middle_targ_text;	// ����м䡰�����赥������
	vector<Rect> rects_down_targ;			// �����һ��3��ͼƬ
	vector<Rect> rects_down_targ_two;		// ����ڶ���3��ͼƬ
	vector<Rect> rects_down_targ_text;		// ������·�ͼƬ������

											//------------------ ģ�ͼ�⣨�ֱ�Դ�ƥ��ͼƬ����������������м�⣩-------------------
	DealModel(mImg, targ_search, targ_up_TuiJian, targ_up, rects_middle_targ, rects_middle_targ_text, rects_down_targ, rects_down_targ_text, rects_down_targ_two);

	//--------------------------------- ��ʾԭ�����ͼƬ�߿� ---------------------------------
	/*for (auto rect : targ_search)
	{
	cv::rectangle(mImg, rect, Scalar(255, 0, 0), 8);
	}*/
	//cv::rectangle(mImg, targ_search, Scalar(0, 255, 0), 8);
	for (auto rect : targ_up_TuiJian)
	{
		cv::rectangle(mImg, rect, Scalar(255, 0, 0), 8);
	}
	cv::rectangle(mImg, targ_up, Scalar(0, 255, 0), 8);
	for (auto rect : rects_middle_targ)
	{
		cv::rectangle(mImg, rect, Scalar(0, 0, 255), 8);
	}
	for (auto rect : rects_middle_targ_text)
	{
		cv::rectangle(mImg, rect, Scalar(255, 0, 0), 8);
	}
	for (auto rect : rects_down_targ)
	{
		cv::rectangle(mImg, rect, Scalar(0, 0, 255), 8);
	}
	for (auto rect : rects_down_targ_text)
	{
		cv::rectangle(mImg, rect, Scalar(255, 0, 0), 8);
	}
	for (auto rect : rects_down_targ_two)
	{
		cv::rectangle(mImg, rect, Scalar(0, 0, 255), 8);
	}
	myshow(mImg, "mImg");		// ��ʾ��ƥ��ͼƬ����Ǳ߿�

								//--------------------------------- �ҵ��Ӿ�����ͼƬ�߿򣬼���ƥ��� ---------------------------------
	float max_match_rate = 0;	// �������ƥ��ͼ

	int index = 0;							// �����Ӿ�������ƥ��ͼƬ����š������±߿�λ����Ϣ
	Rect		 targ_up_detect_index;		//��
	vector<Rect> rects_middle_detect_index; //��
	vector<Rect> down_rects_detect_index;   //��
	Mat img_index;


	for (int i = 0; i < 3; i++)
	{
		cout << i << endl;
		Mat img2 = libImgSrc(rects[i]).clone();			//����������θ����Ӿ���ͼƬ

		double rate = (float)msize.width / img2.cols;	//���Ӿ����е�ͼƬ������С
		resize(img2, img2, Size(), rate, rate);			//���տ�Ƚ��е�����������ƥ��ͼƬһ������£��Ƚϱ߿���Ϣ��

		vector<Rect> targ_up_time_detect;			// ������ʱ���ź�ͼƬ
		vector<Rect> targ_up_search_detect;			// ������������
		vector<Rect> targ_up_text_detect;			// �����桰�Ƽ�������
		Rect		 targ_up_detect;				// �������ͼ
		vector<Rect> rects_middle_detect;			// �м�4��Сͼ
		vector<Rect> rects_middle_text_detect;		// �м䡰�����赥������
		vector<Rect> down_rects_detect;				// �����һ��3��ͼƬ
		vector<Rect> down_rects_text_detect;		// �����һ��3��ͼƬ������Ϣ
		vector<Rect> down_rects_detect_two;			// ����ڶ���3��ͼƬ

		DealDetectImg(img2, targ_up_search_detect, targ_up_text_detect, targ_up_detect, rects_middle_detect, rects_middle_text_detect, down_rects_detect, down_rects_text_detect, down_rects_detect_two);	//���Ӿ�����ͼƬ���λ�ȡ�߿���Ϣ

																																																			//----------------------------------------- ����ƥ�������ʾ��� -----------------------------------------
		float match_up_time = 0, match_up_search = 0, match_up_text = 0, match_middle_text = 0, match_down_text = 0;

		//---------------------------------- ��������λ�ý���ƥ�� ----------------------------------
		for (int j = 0; j<3; j++)
		{
			match_up_search += MatchRate(targ_search[j], targ_up_search_detect[j]);
		}
		match_up_search = match_up_search / 4;
		cout << "������λ��ƥ��ȣ�" << match_up_search << endl;

		//---------------------------------- �ԡ��Ƽ�������λ�ý���ƥ�� ----------------------------------
		for (int j = 0; j<4; j++)
		{
			match_up_text += MatchRate(targ_up_TuiJian[j], targ_up_text_detect[j]);
		}
		match_up_text = match_up_text / 4;
		cout << "��������ƥ��ȣ�" << match_up_text << endl;

		//---------------------------------- ���Ϸ�ͼƬ����ƥ�� -----------------------------------------

		float match_up = MatchRate(targ_up, targ_up_detect);
		cout << "�������ͼƥ��ȣ�" << match_up << endl;

		//---------------------------------- ���м�4��ͼ���������ƥ�� ----------------------------------
		float match_middle = 0;
		for (int j = 0; j<4; j++)
		{
			match_middle += MatchRate(rects_middle_targ[j], rects_middle_detect[j]);
		}
		match_middle = match_middle / 4;
		cout << "�м�4��ÿ�ո�����λ��ƥ�䣺" << match_middle << endl;

		//---------------------------------- �ԡ������������롰���ࡱ����λ�ý���ƥ�� ----------------------------------
		for (int j = 0; j<2; j++)
		{
			match_middle_text += MatchRate(rects_middle_targ_text[j], rects_middle_text_detect[j]);
		}
		match_middle_text = match_middle_text / 2;
		cout << "�����������������λ��ƥ��ȣ�" << match_middle_text << endl;

		//---------------------------------- ������6��ͼ���������ƥ�� ----------------------------------
		float match_down = 0;
		for (int j = 0; j < down_rects_detect.size(); j++)
		{
			float r = MatchRate(down_rects_detect[j], rects_down_targ[j]);
			//			if(r>0.2)
			match_down += r;
		}
		match_down = match_down / down_rects_detect.size();
		cout << "��ƥ�䣺" << match_down << endl;

		// ����ƽ��ƥ��ȣ���ӡ���
		float match_rate = (match_down + match_middle + match_up) / 3;
		cout << "ƽ��ƥ��:" << match_rate << endl;


		//if (match_rate > max_match_rate)		// �ҵ�ƥ��������Ӿ�ͼ����¼�����߿�λ����Ϣ
		//{
		//	max_match_rate = match_rate;
		//	index = i;

		//	targ_up_detect_index = targ_up_detect;			//��
		//	rects_middle_detect_index = rects_middle_detect;//��
		//	down_rects_detect_index = down_rects_detect;	//��
		//	img_index = img2.clone();
		//}

		//--------------------------------- ��ʾ���ͼƬ��� ---------------------------------

		//// ������ʾ�Ӿ�ͼͼƬ��ͬʱ�����Լ��ı߿�
		//cv::rectangle(img2, targ_up_detect, Scalar(0, 0, 255), 8);
		//for (auto rect : down_rects_detect)
		//{
		//	cv::rectangle(img2, rect, Scalar(0, 0, 255), 8);
		//}
		//for (auto rect : rects_middle_detect)
		//{
		//	cv::rectangle(img2, rect, Scalar(0, 0, 255), 8);
		//}

		////������ʾ�Ӿ�ͼ�е�ͼƬ��ͬʱ����ԭ��ƥ��ͼ�εı߿�
		//cv::rectangle(img2, targ_up, Scalar(0, 255, 0), 8);
		//for (auto rect : rects_down_targ)
		//{
		//	cv::rectangle(img2, rect, Scalar(0, 255, 0), 8);
		//}
		//for (auto rect : rects_middle_targ)
		//{
		//	cv::rectangle(img2, rect, Scalar(0, 255, 0), 8);
		//}

		myshow(img2, "img2");
		waitKey(50);
	}

	////�����ƥ��ȵ��Ӿ�ͼ����ʾ�Լ��ı߿�
	//for (auto rect : down_rects_detect_index)
	//{
	//	cv::rectangle(img_index, rect, Scalar(0, 0, 255), 8);
	//}
	//for (auto rect : rects_middle_detect_index)
	//{
	//	cv::rectangle(img_index, rect, Scalar(0, 0, 255), 8);
	//}
	//cv::rectangle(img_index, targ_up_detect_index, Scalar(0, 0, 255), 8);

	////�����ƥ��ȵ��Ӿ�ͼ����ʾԭ����ƥ��ͼƬ�ı߿�
	//for (auto rect : rects_down_targ)
	//{
	//	cv::rectangle(img_index, rect, Scalar(0, 255, 0), 8);
	//}
	//for (auto rect : rects_middle_targ)
	//{
	//	cv::rectangle(img_index, rect, Scalar(0, 255, 0), 8);
	//}
	//cv::rectangle(img_index, targ_up, Scalar(0, 255, 0), 8);
	//myshow(img_index, "img_index");

	waitKey();
}