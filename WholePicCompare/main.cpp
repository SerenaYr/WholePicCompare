#include <iostream>
#include <opencv.hpp>
#include "DealImageUtil.h"
#include <ctime>
#include<fstream>

using namespace cv;
using namespace std;

void myshow(char* name, Mat img) {
	//	cv::namedWindow(name, 0);
	//namedWindow(name, CV_WINDOW_KEEPRATIO);

	imshow(name, img);
}
void myshow(Mat img, char* name) {
	//	cv::namedWindow(name, 0);
	//namedWindow(name, CV_WINDOW_KEEPRATIO);
	double scale = 0.1;
	Size dsize = Size(img.cols*scale, img.rows*scale);
	Mat imgDest = Mat(dsize, CV_32S);
	resize(img, imgDest, dsize);

	imshow(name, imgDest);
}

//ģ�ͼ�⣺�ֱ��ͼ������������������м��
void DealModel(const Mat &src, vector<Rect> &picSearch, vector<Rect> &textTuiJian, Rect &picTuiJian, vector<Rect> &picFourFunc, vector<Rect> &textTuiJianGeDan, vector<Rect> &picTuiJianGeDan,
	vector<Rect> &textXinTuiJian, vector<Rect> &picXinTuiJianLeft, vector<Rect> &picXinTuiJianRight,
	vector<Rect> &textMV, vector<Rect> &picMV, vector<Rect> &textKanDian, vector<Rect> &picKanDian,
	vector<Rect> &textYinYueRen, vector<Rect> &picYinYueRen, vector<Rect> &textPaJian, vector<Rect> &picPaJian,
	vector<Rect> &textTingJianGengDuo, vector<Rect> &picBottonIcon)
{
	Mat mImg = src;

	int width = mImg.cols;
	int height = mImg.rows;

	const Rect rectPicSearch(Point(0, 40), Point(width, 140));		//1.������
	const Rect rectTextTuiJian(Point(30, 140), Point(550, 250));	//2.���ֿ��Ƽ�ſ�俴�㡱����
	const Rect rectPicTuiJian(Point(0, 250), Point(width, 520));	//3.�Ƽ�ͼƬ
	const Rect rectPicFourFunc(Point(0, 520), Point(width, 690));	//4.����30�׵�4�ֹ���ͼƬ
	const Rect rectTextTuiJianGeDan(Point(0, 690), Point(width, 790));	//5.���Ƽ��赥������
	const Rect rectPicTextTuiJianGeDan(Point(0, 790), Point(width, 1550));	//6.�Ƽ�ͼƬ
	const Rect rectTextXinTuiJian(Point(0, 1600), Point(width, 1700));	//7.���¡��Ƽ�������
	const Rect rectPicXinTuiJianLeft(Point(0, 1700), Point(490, 3400));	//8.���¡��Ƽ���ͼƬ��
	const Rect rectPicXinTuiJianRight(Point(491, 1700), Point(width, 3400));//9.���¡��Ƽ���ͼƬ��
	const Rect rectTextMV(Point(0, 3400), Point(width, 3540));				//10.��MV������
	const Rect rectPicMV(Point(0, 3540), Point(width, 4600));				//11.��MV��ͼƬ
	const Rect rectTextKanDian(Point(0, 4650), Point(width, 4780));			//12.�����㡱����
	const Rect rectPicKanDian(Point(0, 4780), Point(width, 5900));			//13.�����㡱ͼƬ
	const Rect rectTextYinYueRen(Point(0, 5930), Point(width, 6070));		//14.�����㡱ͼƬ
	const Rect rectPicYinYueRen(Point(0, 6070), Point(width, 7250));		//15.�����㡱ͼƬ
	const Rect rectTextPaJian(Point(0, 7280), Point(width, 7400));			//16.��ſ�䡱ͼƬ
	const Rect rectPicPaJian(Point(0, 7400), Point(width, 8550));			//17.ſ��ͼƬ
	//const Rect rectTextTingJianGengDuo(Point(0, 8600), Point(width, 9250));	//18.���������ࡱ����
	//const Rect rectPicBottonIcon(Point(0, 9300), Point(width, height));		//19.�ײ�ͼ��

	Mat cannMat;
	cv::Canny(mImg, cannMat, 10, 50);	//��ͼƬ��Canny��Ե���

	//--------------------------------------- 1.�������� ---------------------------------------

	Mat tempPicSearch;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectPicSearch), tempPicSearch, 15);			
	DUtil::RLSA_V(tempPicSearch, tempPicSearch, 15);						
																		
	vector<vector<Point>> countoursPicSearch = DUtil::getCountours(tempPicSearch);	// �����ͨͼ���������

	for (auto countourPicSearch : countoursPicSearch)
	{
		Rect tmpRectPicSearch = cv::boundingRect(countourPicSearch);	// ���Ҿ�������
		tmpRectPicSearch = tmpRectPicSearch + rectPicSearch.tl();		// �������ϽǶ�������
		picSearch.push_back(tmpRectPicSearch);

		cv::rectangle(mImg, tmpRectPicSearch, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(picSearch.begin(), picSearch.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });	

	//--------------------------------------- 2.�����ֿ��Ƽ�ſ�俴�㡱���֡� ---------------------------------------

	Mat tempTextTuiJian;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextTuiJian), tempTextTuiJian, 15);			
	DUtil::RLSA_V(tempTextTuiJian, tempTextTuiJian, 10);						

	vector<vector<Point>> countoursTextTuiJian = DUtil::getCountours(tempTextTuiJian);	// �����ͨͼ���������

	for (auto countourTextTuiJian : countoursTextTuiJian)
	{
		Rect tmpTextTuiJian = cv::boundingRect(countourTextTuiJian);	// ���Ҿ�������
		tmpTextTuiJian = tmpTextTuiJian + rectTextTuiJian.tl();			// �������ϽǶ�������
		textTuiJian.push_back(tmpTextTuiJian);

		cv::rectangle(mImg, tmpTextTuiJian, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(textTuiJian.begin(), textTuiJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 3.���Ƽ�ͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	vector<double>  hproj = DUtil::horizontalProjection(cannMat(rectPicTuiJian));
	vector<double>  vproj = DUtil::verticalProjection(cannMat(rectPicTuiJian));

	vector<int> hds = DUtil::findIndex(hproj, 15);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vector<int> vds = DUtil::findIndex(vproj, 15);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	Point pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	Point pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);
	picTuiJian = Rect(pt1, pt2);						

	picTuiJian.x += rectPicTuiJian.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	picTuiJian.y += rectPicTuiJian.y;

	cv::rectangle(mImg, picTuiJian, Scalar(0, 0, 255), 8);

	//--------------------------------------- 4.������30�׵�4�ֹ���ͼƬ�� ---------------------------------------

	Mat tempFourFunc;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectPicFourFunc), tempFourFunc, 20);
	DUtil::RLSA_V(tempFourFunc, tempFourFunc, 20);

	vector<vector<Point>> countoursFourFunc = DUtil::getCountours(tempFourFunc);	// �����ͨͼ���������

	for (auto countourFourFunc : countoursFourFunc)
	{
		Rect tmpPicFourFunc = cv::boundingRect(countourFourFunc);	// ���Ҿ�������
		tmpPicFourFunc = tmpPicFourFunc + rectPicFourFunc.tl();			// �������ϽǶ�������
		picFourFunc.push_back(tmpPicFourFunc);

		cv::rectangle(mImg, tmpPicFourFunc, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(picFourFunc.begin(), picFourFunc.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 5.�����Ƽ��赥�����֡� ---------------------------------------

	Mat tempTextTuiJianGeDan;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextTuiJianGeDan), tempTextTuiJianGeDan, 20);
	DUtil::RLSA_V(tempTextTuiJianGeDan, tempTextTuiJianGeDan, 10);

	vector<vector<Point>> countoursTextTuiJianGeDan = DUtil::getCountours(tempTextTuiJianGeDan);	// �����ͨͼ���������

	for (auto countourTextTuiJianGeDan : countoursTextTuiJianGeDan)
	{
		Rect tmpTextTuiJianGeDan = cv::boundingRect(countourTextTuiJianGeDan);	// ���Ҿ�������
		tmpTextTuiJianGeDan = tmpTextTuiJianGeDan + rectTextTuiJianGeDan.tl();			// �������ϽǶ�������
		textTuiJianGeDan.push_back(tmpTextTuiJianGeDan);

		cv::rectangle(mImg, tmpTextTuiJianGeDan, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(textTuiJianGeDan.begin(), textTuiJianGeDan.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 6.���Ƽ�ͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicTextTuiJianGeDan));
	vproj = DUtil::verticalProjection(cannMat(rectPicTextTuiJianGeDan));
	
	double th = rectPicTextTuiJianGeDan.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);
	
	Rect precisePicTuiJianGeDan;
	precisePicTuiJianGeDan = Rect(pt1, pt2);

	precisePicTuiJianGeDan.x += rectPicTextTuiJianGeDan.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicTuiJianGeDan.y += rectPicTextTuiJianGeDan.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicTuiJianGeDan;
	DUtil::RLSA_H(cannMat(precisePicTuiJianGeDan), tempPicTuiJianGeDan, 30);	// �����ؾ���С��5�����ص��������
	DUtil::RLSA_V(tempPicTuiJianGeDan, tempPicTuiJianGeDan, 15);

	vector<vector<Point>> countoursPicTuiJianGeDan = DUtil::getCountours(tempPicTuiJianGeDan);	// �����ͨͼ���������

	for (auto countourPicTuiJianGeDan : countoursPicTuiJianGeDan)
	{
		Rect tmpPicTuiJianGeDan = cv::boundingRect(countourPicTuiJianGeDan);	// ���Ҿ�������
		tmpPicTuiJianGeDan = tmpPicTuiJianGeDan + precisePicTuiJianGeDan.tl();			// �������ϽǶ�������
		picTuiJianGeDan.push_back(tmpPicTuiJianGeDan);

		cv::rectangle(mImg, tmpPicTuiJianGeDan, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picTuiJianGeDan.begin(), picTuiJianGeDan.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 7.�����¡��Ƽ������֡� ---------------------------------------

	Mat tempTextXinTuiJian;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextXinTuiJian), tempTextXinTuiJian, 20);
	DUtil::RLSA_V(tempTextXinTuiJian, tempTextXinTuiJian, 10);

	vector<vector<Point>> countoursTextXinTuiJian = DUtil::getCountours(tempTextXinTuiJian);	// �����ͨͼ���������

	for (auto countourTextXinTuiJian : countoursTextXinTuiJian)
	{
		Rect tmpTextXinTuiJian = cv::boundingRect(countourTextXinTuiJian);	// ���Ҿ�������
		tmpTextXinTuiJian = tmpTextXinTuiJian + rectTextXinTuiJian.tl();			// �������ϽǶ�������
		textXinTuiJian.push_back(tmpTextXinTuiJian);

		cv::rectangle(mImg, tmpTextXinTuiJian, Scalar(0, 0, 255), 8);
	}

	//--------------------------------------- 8.�����¡��Ƽ�ͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicXinTuiJianLeft));
	vproj = DUtil::verticalProjection(cannMat(rectPicXinTuiJianLeft));

	th = rectPicXinTuiJianLeft.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicXinTuiJianLeft;
	precisePicXinTuiJianLeft = Rect(pt1, pt2);

	precisePicXinTuiJianLeft.x += rectPicXinTuiJianLeft.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicXinTuiJianLeft.y += rectPicXinTuiJianLeft.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicXinTuiJianLeft;
	DUtil::RLSA_H(cannMat(precisePicXinTuiJianLeft), tempPicXinTuiJianLeft, 12);
	DUtil::RLSA_V(tempPicXinTuiJianLeft, tempPicXinTuiJianLeft, 10);

	vector<vector<Point>> countoursPicXinTuiJianLeft = DUtil::getCountours(tempPicXinTuiJianLeft);	// �����ͨͼ���������

	for (auto countourPicXinTuiJianLeft : countoursPicXinTuiJianLeft)
	{
		Rect tmpPicXinTuiJianLeft = cv::boundingRect(countourPicXinTuiJianLeft);	// ���Ҿ�������
		tmpPicXinTuiJianLeft = tmpPicXinTuiJianLeft + precisePicXinTuiJianLeft.tl();			// �������ϽǶ�������
		picXinTuiJianLeft.push_back(tmpPicXinTuiJianLeft);

		cv::rectangle(mImg, tmpPicXinTuiJianLeft, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picXinTuiJianLeft.begin(), picXinTuiJianLeft.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 9.�����¡��Ƽ�ͼƬ�ҡ� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicXinTuiJianRight));
	vproj = DUtil::verticalProjection(cannMat(rectPicXinTuiJianRight));

	th = rectPicXinTuiJianRight.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicXinTuiJianRight = Rect(pt1, pt2);

	precisePicXinTuiJianRight.x += rectPicXinTuiJianRight.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicXinTuiJianRight.y += rectPicXinTuiJianRight.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicXinTuiJianRight;
	DUtil::RLSA_H(cannMat(precisePicXinTuiJianRight), tempPicXinTuiJianRight, 12);
	DUtil::RLSA_V(tempPicXinTuiJianRight, tempPicXinTuiJianRight, 15);

	vector<vector<Point>> countoursPicXinTuiJianRight = DUtil::getCountours(tempPicXinTuiJianRight);	// �����ͨͼ���������

	for (auto countourPicXinTuiJianRight : countoursPicXinTuiJianRight)
	{
		Rect tmpPicXinTuiJianRight = cv::boundingRect(countourPicXinTuiJianRight);	// ���Ҿ�������
		tmpPicXinTuiJianRight = tmpPicXinTuiJianRight + precisePicXinTuiJianRight.tl();			// �������ϽǶ�������
		picXinTuiJianRight.push_back(tmpPicXinTuiJianRight);

		cv::rectangle(mImg, tmpPicXinTuiJianRight, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picXinTuiJianRight.begin(), picXinTuiJianRight.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 10.����MV�����֡� ---------------------------------------

	Mat tempTextMV;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextMV), tempTextMV, 20);
	DUtil::RLSA_V(tempTextMV, tempTextMV, 10);

	vector<vector<Point>> countoursTextMV = DUtil::getCountours(tempTextMV);	// �����ͨͼ���������

	for (auto countourTextMV : countoursTextMV)
	{
		Rect tmpTextMV = cv::boundingRect(countourTextMV);	// ���Ҿ�������
		tmpTextMV = tmpTextMV + rectTextMV.tl();			// �������ϽǶ�������
		textMV.push_back(tmpTextMV);

		cv::rectangle(mImg, tmpTextMV, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(textMV.begin(), textMV.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 11.��MVͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicMV));
	vproj = DUtil::verticalProjection(cannMat(rectPicMV));

	th = rectPicMV.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicMV;
	precisePicMV = Rect(pt1, pt2);

	precisePicMV.x += rectPicMV.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicMV.y += rectPicMV.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicMV;
	DUtil::RLSA_H(cannMat(precisePicMV), tempPicMV, 15);
	DUtil::RLSA_V(tempPicMV, tempPicMV, 10);

	vector<vector<Point>> countoursPicMV = DUtil::getCountours(tempPicMV);	// �����ͨͼ���������

	for (auto countourPicMV : countoursPicMV)
	{
		Rect tmpPicMV = cv::boundingRect(countourPicMV);	// ���Ҿ�������
		tmpPicMV = tmpPicMV + precisePicMV.tl();			// �������ϽǶ�������
		picMV.push_back(tmpPicMV);

		cv::rectangle(mImg, tmpPicMV, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picMV.begin(), picMV.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 12.�������㡱���֡� ---------------------------------------

	Mat tempTextKanDian;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextKanDian), tempTextKanDian, 20);
	DUtil::RLSA_V(tempTextKanDian, tempTextKanDian, 10);

	vector<vector<Point>> countoursTextKanDian = DUtil::getCountours(tempTextKanDian);	// �����ͨͼ���������

	for (auto countourTextKanDian : countoursTextKanDian)
	{
		Rect tmpTextKanDian = cv::boundingRect(countourTextKanDian);	// ���Ҿ�������
		tmpTextKanDian = tmpTextKanDian + rectTextKanDian.tl();			// �������ϽǶ�������
		textKanDian.push_back(tmpTextKanDian);

		cv::rectangle(mImg, tmpTextKanDian, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(textKanDian.begin(), textKanDian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 13.������ͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicKanDian));
	vproj = DUtil::verticalProjection(cannMat(rectPicKanDian));

	th = rectPicKanDian.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th*0.1);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicKanDian = Rect(pt1, pt2);

	precisePicKanDian.x += rectPicKanDian.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicKanDian.y += rectPicKanDian.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicKanDian;
	DUtil::RLSA_H(cannMat(precisePicKanDian), tempPicKanDian, 15);
	DUtil::RLSA_V(tempPicKanDian, tempPicKanDian, 15);

	vector<vector<Point>> countoursPicKanDian = DUtil::getCountours(tempPicKanDian);	// �����ͨͼ���������

	for (auto countourPicKanDian : countoursPicKanDian)
	{
		Rect tmpPicKanDian = cv::boundingRect(countourPicKanDian);	// ���Ҿ�������
		tmpPicKanDian = tmpPicKanDian + precisePicKanDian.tl();		// �������ϽǶ�������
		picKanDian.push_back(tmpPicKanDian);

		if (tmpPicKanDian.width < 35)
			continue;

		cv::rectangle(mImg, tmpPicKanDian, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picKanDian.begin(), picKanDian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 14.���������ˡ����֡� ---------------------------------------

	Mat tempTextYinYueRen;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextYinYueRen), tempTextYinYueRen, 20);
	DUtil::RLSA_V(tempTextYinYueRen, tempTextYinYueRen, 10);

	vector<vector<Point>> countoursTextYinYueRen = DUtil::getCountours(tempTextYinYueRen);	// �����ͨͼ���������

	for (auto countourTextYinYueRen : countoursTextYinYueRen)
	{
		Rect tmpTextYinYueRen = cv::boundingRect(countourTextYinYueRen);	// ���Ҿ�������
		tmpTextYinYueRen = tmpTextYinYueRen + rectTextYinYueRen.tl();			// �������ϽǶ�������
		textYinYueRen.push_back(tmpTextYinYueRen);

		cv::rectangle(mImg, tmpTextYinYueRen, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(textYinYueRen.begin(), textYinYueRen.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 15.��������ͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicYinYueRen));
	vproj = DUtil::verticalProjection(cannMat(rectPicYinYueRen));

	th = rectPicYinYueRen.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicYinYueRen = Rect(pt1, pt2);

	precisePicYinYueRen.x += rectPicYinYueRen.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicYinYueRen.y += rectPicYinYueRen.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicYinYueRen;
	DUtil::RLSA_H(cannMat(precisePicYinYueRen), tempPicYinYueRen, 12);
	DUtil::RLSA_V(tempPicYinYueRen, tempPicYinYueRen, 15);

	vector<vector<Point>> countoursPicYinYueRen = DUtil::getCountours(tempPicYinYueRen);	// �����ͨͼ���������

	for (auto countourPicYinYueRen : countoursPicYinYueRen)
	{
		Rect tmpPicYinYueRen = cv::boundingRect(countourPicYinYueRen);	// ���Ҿ�������
		tmpPicYinYueRen = tmpPicYinYueRen + precisePicYinYueRen.tl();			// �������ϽǶ�������
		picYinYueRen.push_back(tmpPicYinYueRen);

		cv::rectangle(mImg, tmpPicYinYueRen, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picYinYueRen.begin(), picYinYueRen.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 16.����ſ�䡱���֡� ---------------------------------------

	Mat tempTextPaJian;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextPaJian), tempTextPaJian, 20);
	DUtil::RLSA_V(tempTextPaJian, tempTextPaJian, 10);

	vector<vector<Point>> countoursTextPaJian = DUtil::getCountours(tempTextPaJian);	// �����ͨͼ���������

	for (auto countourTextPaJian : countoursTextPaJian)
	{
		Rect tmpTextPaJian = cv::boundingRect(countourTextPaJian);	// ���Ҿ�������
		tmpTextPaJian = tmpTextPaJian + rectTextPaJian.tl();			// �������ϽǶ�������
		textPaJian.push_back(tmpTextPaJian);

		cv::rectangle(mImg, tmpTextPaJian, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(textPaJian.begin(), textPaJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 17.��ſ��ͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicPaJian));
	vproj = DUtil::verticalProjection(cannMat(rectPicPaJian));

	th = rectPicPaJian.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicPaJian = Rect(pt1, pt2);

	precisePicPaJian.x += rectPicPaJian.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicPaJian.y += rectPicPaJian.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicPaJian;
	DUtil::RLSA_H(cannMat(precisePicPaJian), tempPicPaJian, 15);
	DUtil::RLSA_V(tempPicPaJian, tempPicPaJian, 50);

	vector<vector<Point>> countoursPicPaJian = DUtil::getCountours(tempPicPaJian);	// �����ͨͼ���������

	for (auto countourPicPaJian : countoursPicPaJian)
	{
		Rect tmpPicPaJian = cv::boundingRect(countourPicPaJian);	// ���Ҿ�������
		tmpPicPaJian = tmpPicPaJian + precisePicPaJian.tl();			// �������ϽǶ�������
		picPaJian.push_back(tmpPicPaJian);

		cv::rectangle(mImg, tmpPicPaJian, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picPaJian.begin(), picPaJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	////--------------------------------------- 18.�����������ࡱ���֡� ---------------------------------------

	//Mat tempTextTingJianGengDuo;

	//// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	//DUtil::RLSA_H(cannMat(rectTextTingJianGengDuo), tempTextTingJianGengDuo, 100);
	//DUtil::RLSA_V(tempTextTingJianGengDuo, tempTextTingJianGengDuo, 10);

	//vector<vector<Point>> countoursTextTingJianGengDuo = DUtil::getCountours(tempTextTingJianGengDuo);	// �����ͨͼ���������

	//for (auto countourTextTingJianGengDuo : countoursTextTingJianGengDuo)
	//{
	//	Rect tmpTextTingJianGengDuo = cv::boundingRect(countourTextTingJianGengDuo);	// ���Ҿ�������
	//	tmpTextTingJianGengDuo = tmpTextTingJianGengDuo + rectTextTingJianGengDuo.tl();			// �������ϽǶ�������
	//	textTingJianGengDuo.push_back(tmpTextTingJianGengDuo);

	//	cv::rectangle(mImg, tmpTextTingJianGengDuo, Scalar(0, 0, 255), 8);
	//}

	//// ����������ʼ��xλ�ö�Rect�����������
	//sort(textTingJianGengDuo.begin(), textTingJianGengDuo.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 19.���ײ�5��ͼ�꡿ ---------------------------------------

	//Mat tempBottonIcon;

	//// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	//DUtil::RLSA_H(cannMat(rectPicBottonIcon), tempBottonIcon, 20);
	//DUtil::RLSA_V(tempBottonIcon, tempBottonIcon, 20);

	//vector<vector<Point>> countoursBottonIcon = DUtil::getCountours(tempBottonIcon);	// �����ͨͼ���������

	//for (auto countourBottonIcon : countoursBottonIcon)
	//{
	//	Rect tmpPicBottonIcon = cv::boundingRect(countourBottonIcon);	// ���Ҿ�������
	//	tmpPicBottonIcon = tmpPicBottonIcon + rectPicBottonIcon.tl();			// �������ϽǶ�������
	//	picBottonIcon.push_back(tmpPicBottonIcon);

	//	cv::rectangle(mImg, tmpPicBottonIcon, Scalar(255, 0, 0), 8);
	//}

	//// ����������ʼ��xλ�ö�Rect�����������
	//sort(picBottonIcon.begin(), picBottonIcon.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });


	waitKey();
}

//ģ�ͼ�⣺�ֱ��ͼ������������������м��
void DealModel2(const Mat &src, vector<Rect> &picSearch, vector<Rect> &textTuiJian, Rect &picTuiJian, vector<Rect> picFourFunc, vector<Rect> textTuiJianGeDan, vector<Rect> picTuiJianGeDan,
	vector<Rect> textXinTuiJian, vector<Rect> picXinTuiJianLeft, vector<Rect> picXinTuiJianRight,
	vector<Rect> textMV, vector<Rect> picMV, vector<Rect> textKanDian, vector<Rect> picKanDian,
	vector<Rect> textYinYueRen, vector<Rect> picYinYueRen, vector<Rect> textPaJian, vector<Rect> picPaJian,
	vector<Rect> textTingJianGengDuo, vector<Rect> picBottonIcon)
{
	Mat mImg = src;

	int width = mImg.cols;
	int height = mImg.rows;

	const Rect rectPicSearch(Point(0, 40), Point(width, 140));		//1.������
	const Rect rectTextTuiJian(Point(0, 140), Point(550, 250));	//2.���ֿ��Ƽ�ſ�俴�㡱����
	const Rect rectPicTuiJian(Point(0, 250), Point(width, 520));	//3.�Ƽ�ͼƬ
	const Rect rectPicFourFunc(Point(0, 520), Point(width, 690));	//4.����30�׵�4�ֹ���ͼƬ
	const Rect rectTextTuiJianGeDan(Point(0, 690), Point(width, 790));	//5.���Ƽ��赥������
	const Rect rectPicTextTuiJianGeDan(Point(0, 790), Point(width, 1550));	//6.�Ƽ�ͼƬ
	const Rect rectTextXinTuiJian(Point(0, 1600), Point(width, 1700));	//7.���¡��Ƽ�������
	const Rect rectPicXinTuiJianLeft(Point(0, 1700), Point(490, 3400));	//8.���¡��Ƽ���ͼƬ��
	const Rect rectPicXinTuiJianRight(Point(491, 1700), Point(width, 3400));//9.���¡��Ƽ���ͼƬ��
	const Rect rectTextMV(Point(0, 3400), Point(width, 3540));				//10.��MV������
	const Rect rectPicMV(Point(0, 3540), Point(width, 4600));				//11.��MV��ͼƬ
	const Rect rectTextKanDian(Point(0, 4650), Point(width, 4780));			//12.�����㡱����
	const Rect rectPicKanDian(Point(0, 4780), Point(width, 5900));			//13.�����㡱ͼƬ
	const Rect rectTextYinYueRen(Point(0, 5930), Point(width, 6070));		//14.�����㡱ͼƬ
	const Rect rectPicYinYueRen(Point(0, 6070), Point(width, 7250));		//15.�����㡱ͼƬ
	const Rect rectTextPaJian(Point(0, 7280), Point(width, 7400));			//16.��ſ�䡱ͼƬ
	const Rect rectPicPaJian(Point(0, 7400), Point(width, 8550));			//17.ſ��ͼƬ
																			//const Rect rectTextTingJianGengDuo(Point(0, 8600), Point(width, 9250));	//18.���������ࡱ����
																			//const Rect rectPicBottonIcon(Point(0, 9300), Point(width, height));		//19.�ײ�ͼ��

	Mat cannMat;
	cv::Canny(mImg, cannMat, 10, 50);	//��ͼƬ��Canny��Ե���

	//--------------------------------------- 1.�������� ---------------------------------------

	Mat tempPicSearch;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectPicSearch), tempPicSearch, 15);
	DUtil::RLSA_V(tempPicSearch, tempPicSearch, 15);

	vector<vector<Point>> countoursPicSearch = DUtil::getCountours(tempPicSearch);	// �����ͨͼ���������

	for (auto countourPicSearch : countoursPicSearch)
	{
		Rect tmpRectPicSearch = cv::boundingRect(countourPicSearch);	// ���Ҿ�������
		tmpRectPicSearch = tmpRectPicSearch + rectPicSearch.tl();		// �������ϽǶ�������
		picSearch.push_back(tmpRectPicSearch);

		cv::rectangle(mImg, tmpRectPicSearch, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(picSearch.begin(), picSearch.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 2.�����ֿ��Ƽ�ſ�俴�㡱���֡� ---------------------------------------

	Mat tempTextTuiJian;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextTuiJian), tempTextTuiJian, 10);
	DUtil::RLSA_V(tempTextTuiJian, tempTextTuiJian, 10);

	vector<vector<Point>> countoursTextTuiJian = DUtil::getCountours(tempTextTuiJian);	// �����ͨͼ���������

	for (auto countourTextTuiJian : countoursTextTuiJian)
	{
		Rect tmpTextTuiJian = cv::boundingRect(countourTextTuiJian);	// ���Ҿ�������
		tmpTextTuiJian = tmpTextTuiJian + rectTextTuiJian.tl();			// �������ϽǶ�������
		textTuiJian.push_back(tmpTextTuiJian);

		cv::rectangle(mImg, tmpTextTuiJian, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(textTuiJian.begin(), textTuiJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 3.���Ƽ�ͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	vector<double>  hproj = DUtil::horizontalProjection(cannMat(rectPicTuiJian));
	vector<double>  vproj = DUtil::verticalProjection(cannMat(rectPicTuiJian));

	vector<int> hds = DUtil::findIndex(hproj, 15);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vector<int> vds = DUtil::findIndex(vproj, 15);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	Point pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	Point pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);
	picTuiJian = Rect(pt1, pt2);

	picTuiJian.x += rectPicTuiJian.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	picTuiJian.y += rectPicTuiJian.y;

	cv::rectangle(mImg, picTuiJian, Scalar(0, 0, 255), 8);

	//--------------------------------------- 4.������30�׵�4�ֹ���ͼƬ�� ---------------------------------------

	Mat tempFourFunc;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectPicFourFunc), tempFourFunc, 20);
	DUtil::RLSA_V(tempFourFunc, tempFourFunc, 20);

	vector<vector<Point>> countoursFourFunc = DUtil::getCountours(tempFourFunc);	// �����ͨͼ���������

	for (auto countourFourFunc : countoursFourFunc)
	{
		Rect tmpPicFourFunc = cv::boundingRect(countourFourFunc);	// ���Ҿ�������
		tmpPicFourFunc = tmpPicFourFunc + rectPicFourFunc.tl();			// �������ϽǶ�������
		picFourFunc.push_back(tmpPicFourFunc);

		//cv::rectangle(mImg, tmpPicFourFunc, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(picFourFunc.begin(), picFourFunc.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 5.�����Ƽ��赥�����֡� ---------------------------------------

	Mat tempTextTuiJianGeDan;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextTuiJianGeDan), tempTextTuiJianGeDan, 20);
	DUtil::RLSA_V(tempTextTuiJianGeDan, tempTextTuiJianGeDan, 20);

	vector<vector<Point>> countoursTextTuiJianGeDan = DUtil::getCountours(tempTextTuiJianGeDan);	// �����ͨͼ���������

	for (auto countourTextTuiJianGeDan : countoursTextTuiJianGeDan)
	{
		Rect tmpTextTuiJianGeDan = cv::boundingRect(countourTextTuiJianGeDan);	// ���Ҿ�������
		tmpTextTuiJianGeDan = tmpTextTuiJianGeDan + rectTextTuiJianGeDan.tl();			// �������ϽǶ�������
		textTuiJianGeDan.push_back(tmpTextTuiJianGeDan);

		cv::rectangle(mImg, tmpTextTuiJianGeDan, Scalar(0, 0, 255), 8);
	}
	cv::rectangle(mImg, Rect(Point(44,724),Point(44+189,724+45)), Scalar(255, 0, 0), 8);
	cv::rectangle(mImg, Rect(Point(569, 743), Point(569+142, 743+26)), Scalar(255, 0, 0), 8);

	// ����������ʼ��xλ�ö�Rect�����������
	sort(textTuiJianGeDan.begin(), textTuiJianGeDan.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 6.���Ƽ�ͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicTextTuiJianGeDan));
	vproj = DUtil::verticalProjection(cannMat(rectPicTextTuiJianGeDan));

	double th = rectPicTextTuiJianGeDan.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicTuiJianGeDan;
	precisePicTuiJianGeDan = Rect(pt1, pt2);

	precisePicTuiJianGeDan.x += rectPicTextTuiJianGeDan.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicTuiJianGeDan.y += rectPicTextTuiJianGeDan.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicTuiJianGeDan;
	DUtil::RLSA_H(cannMat(precisePicTuiJianGeDan), tempPicTuiJianGeDan, 30);	// �����ؾ���С��5�����ص��������
	DUtil::RLSA_V(tempPicTuiJianGeDan, tempPicTuiJianGeDan, 15);

	vector<vector<Point>> countoursPicTuiJianGeDan = DUtil::getCountours(tempPicTuiJianGeDan);	// �����ͨͼ���������

	for (auto countourPicTuiJianGeDan : countoursPicTuiJianGeDan)
	{
		Rect tmpPicTuiJianGeDan = cv::boundingRect(countourPicTuiJianGeDan);	// ���Ҿ�������
		tmpPicTuiJianGeDan = tmpPicTuiJianGeDan + precisePicTuiJianGeDan.tl();			// �������ϽǶ�������
		picTuiJianGeDan.push_back(tmpPicTuiJianGeDan);

		//cv::rectangle(mImg, tmpPicTuiJianGeDan, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picTuiJianGeDan.begin(), picTuiJianGeDan.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 7.�����¡��Ƽ������֡� ---------------------------------------

	Mat tempTextXinTuiJian;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextXinTuiJian), tempTextXinTuiJian, 20);
	DUtil::RLSA_V(tempTextXinTuiJian, tempTextXinTuiJian, 10);

	vector<vector<Point>> countoursTextXinTuiJian = DUtil::getCountours(tempTextXinTuiJian);	// �����ͨͼ���������

	for (auto countourTextXinTuiJian : countoursTextXinTuiJian)
	{
		Rect tmpTextXinTuiJian = cv::boundingRect(countourTextXinTuiJian);	// ���Ҿ�������
		tmpTextXinTuiJian = tmpTextXinTuiJian + rectTextXinTuiJian.tl();			// �������ϽǶ�������
		textXinTuiJian.push_back(tmpTextXinTuiJian);

		cv::rectangle(mImg, tmpTextXinTuiJian, Scalar(0, 0, 255), 8);
	}

	//--------------------------------------- 8.�����¡��Ƽ�ͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicXinTuiJianLeft));
	vproj = DUtil::verticalProjection(cannMat(rectPicXinTuiJianLeft));

	th = rectPicXinTuiJianLeft.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicXinTuiJianLeft;
	precisePicXinTuiJianLeft = Rect(pt1, pt2);

	precisePicXinTuiJianLeft.x += rectPicXinTuiJianLeft.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicXinTuiJianLeft.y += rectPicXinTuiJianLeft.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicXinTuiJianLeft;
	DUtil::RLSA_H(cannMat(precisePicXinTuiJianLeft), tempPicXinTuiJianLeft, 12);
	DUtil::RLSA_V(tempPicXinTuiJianLeft, tempPicXinTuiJianLeft, 10);

	vector<vector<Point>> countoursPicXinTuiJianLeft = DUtil::getCountours(tempPicXinTuiJianLeft);	// �����ͨͼ���������

	for (auto countourPicXinTuiJianLeft : countoursPicXinTuiJianLeft)
	{
		Rect tmpPicXinTuiJianLeft = cv::boundingRect(countourPicXinTuiJianLeft);	// ���Ҿ�������
		tmpPicXinTuiJianLeft = tmpPicXinTuiJianLeft + precisePicXinTuiJianLeft.tl();			// �������ϽǶ�������
		picXinTuiJianLeft.push_back(tmpPicXinTuiJianLeft);

		cv::rectangle(mImg, tmpPicXinTuiJianLeft, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picXinTuiJianLeft.begin(), picXinTuiJianLeft.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 9.�����¡��Ƽ�ͼƬ�ҡ� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicXinTuiJianRight));
	vproj = DUtil::verticalProjection(cannMat(rectPicXinTuiJianRight));

	th = rectPicXinTuiJianRight.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicXinTuiJianRight = Rect(pt1, pt2);

	precisePicXinTuiJianRight.x += rectPicXinTuiJianRight.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicXinTuiJianRight.y += rectPicXinTuiJianRight.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicXinTuiJianRight;
	DUtil::RLSA_H(cannMat(precisePicXinTuiJianRight), tempPicXinTuiJianRight, 12);
	DUtil::RLSA_V(tempPicXinTuiJianRight, tempPicXinTuiJianRight, 15);

	vector<vector<Point>> countoursPicXinTuiJianRight = DUtil::getCountours(tempPicXinTuiJianRight);	// �����ͨͼ���������

	for (auto countourPicXinTuiJianRight : countoursPicXinTuiJianRight)
	{
		Rect tmpPicXinTuiJianRight = cv::boundingRect(countourPicXinTuiJianRight);	// ���Ҿ�������
		tmpPicXinTuiJianRight = tmpPicXinTuiJianRight + precisePicXinTuiJianRight.tl();			// �������ϽǶ�������
		picXinTuiJianRight.push_back(tmpPicXinTuiJianRight);

		cv::rectangle(mImg, tmpPicXinTuiJianRight, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picXinTuiJianRight.begin(), picXinTuiJianRight.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 10.����MV�����֡� ---------------------------------------

	Mat tempTextMV;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextMV), tempTextMV, 20);
	DUtil::RLSA_V(tempTextMV, tempTextMV, 10);

	vector<vector<Point>> countoursTextMV = DUtil::getCountours(tempTextMV);	// �����ͨͼ���������

	for (auto countourTextMV : countoursTextMV)
	{
		Rect tmpTextMV = cv::boundingRect(countourTextMV);	// ���Ҿ�������
		tmpTextMV = tmpTextMV + rectTextMV.tl();			// �������ϽǶ�������
		textMV.push_back(tmpTextMV);

		cv::rectangle(mImg, tmpTextMV, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(textMV.begin(), textMV.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 11.��MVͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicMV));
	vproj = DUtil::verticalProjection(cannMat(rectPicMV));

	th = rectPicMV.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicMV;
	precisePicMV = Rect(pt1, pt2);

	precisePicMV.x += rectPicMV.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicMV.y += rectPicMV.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicMV;
	DUtil::RLSA_H(cannMat(precisePicMV), tempPicMV, 15);
	DUtil::RLSA_V(tempPicMV, tempPicMV, 10);

	vector<vector<Point>> countoursPicMV = DUtil::getCountours(tempPicMV);	// �����ͨͼ���������

	for (auto countourPicMV : countoursPicMV)
	{
		Rect tmpPicMV = cv::boundingRect(countourPicMV);	// ���Ҿ�������
		tmpPicMV = tmpPicMV + precisePicMV.tl();			// �������ϽǶ�������
		picMV.push_back(tmpPicMV);

		cv::rectangle(mImg, tmpPicMV, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picMV.begin(), picMV.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 12.�������㡱���֡� ---------------------------------------

	Mat tempTextKanDian;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextKanDian), tempTextKanDian, 20);
	DUtil::RLSA_V(tempTextKanDian, tempTextKanDian, 10);

	vector<vector<Point>> countoursTextKanDian = DUtil::getCountours(tempTextKanDian);	// �����ͨͼ���������

	for (auto countourTextKanDian : countoursTextKanDian)
	{
		Rect tmpTextKanDian = cv::boundingRect(countourTextKanDian);	// ���Ҿ�������
		tmpTextKanDian = tmpTextKanDian + rectTextKanDian.tl();			// �������ϽǶ�������
		textKanDian.push_back(tmpTextKanDian);

		cv::rectangle(mImg, tmpTextKanDian, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(textKanDian.begin(), textKanDian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 13.������ͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicKanDian));
	vproj = DUtil::verticalProjection(cannMat(rectPicKanDian));

	th = rectPicKanDian.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th*0.1);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicKanDian = Rect(pt1, pt2);

	precisePicKanDian.x += rectPicKanDian.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicKanDian.y += rectPicKanDian.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicKanDian;
	DUtil::RLSA_H(cannMat(precisePicKanDian), tempPicKanDian, 15);
	DUtil::RLSA_V(tempPicKanDian, tempPicKanDian, 15);

	vector<vector<Point>> countoursPicKanDian = DUtil::getCountours(tempPicKanDian);	// �����ͨͼ���������

	for (auto countourPicKanDian : countoursPicKanDian)
	{
		Rect tmpPicKanDian = cv::boundingRect(countourPicKanDian);	// ���Ҿ�������
		tmpPicKanDian = tmpPicKanDian + precisePicKanDian.tl();		// �������ϽǶ�������
		picKanDian.push_back(tmpPicKanDian);

		if (tmpPicKanDian.width < 35)
			continue;

		cv::rectangle(mImg, tmpPicKanDian, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picKanDian.begin(), picKanDian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 14.���������ˡ����֡� ---------------------------------------

	Mat tempTextYinYueRen;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextYinYueRen), tempTextYinYueRen, 20);
	DUtil::RLSA_V(tempTextYinYueRen, tempTextYinYueRen, 10);

	vector<vector<Point>> countoursTextYinYueRen = DUtil::getCountours(tempTextYinYueRen);	// �����ͨͼ���������

	for (auto countourTextYinYueRen : countoursTextYinYueRen)
	{
		Rect tmpTextYinYueRen = cv::boundingRect(countourTextYinYueRen);	// ���Ҿ�������
		tmpTextYinYueRen = tmpTextYinYueRen + rectTextYinYueRen.tl();			// �������ϽǶ�������
		textYinYueRen.push_back(tmpTextYinYueRen);

		cv::rectangle(mImg, tmpTextYinYueRen, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(textYinYueRen.begin(), textYinYueRen.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 15.��������ͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicYinYueRen));
	vproj = DUtil::verticalProjection(cannMat(rectPicYinYueRen));

	th = rectPicYinYueRen.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicYinYueRen = Rect(pt1, pt2);

	precisePicYinYueRen.x += rectPicYinYueRen.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicYinYueRen.y += rectPicYinYueRen.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicYinYueRen;
	DUtil::RLSA_H(cannMat(precisePicYinYueRen), tempPicYinYueRen, 12);
	DUtil::RLSA_V(tempPicYinYueRen, tempPicYinYueRen, 15);

	vector<vector<Point>> countoursPicYinYueRen = DUtil::getCountours(tempPicYinYueRen);	// �����ͨͼ���������

	for (auto countourPicYinYueRen : countoursPicYinYueRen)
	{
		Rect tmpPicYinYueRen = cv::boundingRect(countourPicYinYueRen);	// ���Ҿ�������
		tmpPicYinYueRen = tmpPicYinYueRen + precisePicYinYueRen.tl();			// �������ϽǶ�������
		picYinYueRen.push_back(tmpPicYinYueRen);

		cv::rectangle(mImg, tmpPicYinYueRen, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picYinYueRen.begin(), picYinYueRen.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 16.����ſ�䡱���֡� ---------------------------------------

	Mat tempTextPaJian;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	DUtil::RLSA_H(cannMat(rectTextPaJian), tempTextPaJian, 20);
	DUtil::RLSA_V(tempTextPaJian, tempTextPaJian, 10);

	vector<vector<Point>> countoursTextPaJian = DUtil::getCountours(tempTextPaJian);	// �����ͨͼ���������

	for (auto countourTextPaJian : countoursTextPaJian)
	{
		Rect tmpTextPaJian = cv::boundingRect(countourTextPaJian);	// ���Ҿ�������
		tmpTextPaJian = tmpTextPaJian + rectTextPaJian.tl();			// �������ϽǶ�������
		textPaJian.push_back(tmpTextPaJian);

		cv::rectangle(mImg, tmpTextPaJian, Scalar(0, 0, 255), 8);
	}

	// ����������ʼ��xλ�ö�Rect�����������
	sort(textPaJian.begin(), textPaJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 17.��ſ��ͼƬ�� ---------------------------------------

	//ˮƽ��ֱͶӰ
	hproj = DUtil::horizontalProjection(cannMat(rectPicPaJian));
	vproj = DUtil::verticalProjection(cannMat(rectPicPaJian));

	th = rectPicPaJian.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������
	vds = DUtil::findIndex(vproj, th);	//���ұ߽磬�����г���15�����ص㿪ʼ��Ϊͼ������

	pt1 = Point(hds[0], vds[0]);				//ͨ��ͶӰ���ص㳬���ٽ�ֵ��ȷ��׼ȷͼ��Χ
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicPaJian = Rect(pt1, pt2);

	precisePicPaJian.x += rectPicPaJian.x;	//ƫ�ƣ�ʶ���׼ȷλ��
	precisePicPaJian.y += rectPicPaJian.y;

	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	Mat tempPicPaJian;
	DUtil::RLSA_H(cannMat(precisePicPaJian), tempPicPaJian, 15);
	DUtil::RLSA_V(tempPicPaJian, tempPicPaJian, 50);

	vector<vector<Point>> countoursPicPaJian = DUtil::getCountours(tempPicPaJian);	// �����ͨͼ���������

	for (auto countourPicPaJian : countoursPicPaJian)
	{
		Rect tmpPicPaJian = cv::boundingRect(countourPicPaJian);	// ���Ҿ�������
		tmpPicPaJian = tmpPicPaJian + precisePicPaJian.tl();			// �������ϽǶ�������
		picPaJian.push_back(tmpPicPaJian);

		cv::rectangle(mImg, tmpPicPaJian, Scalar(0, 0, 255), 8);
	}

	//����������
	sort(picPaJian.begin(), picPaJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	////--------------------------------------- 18.�����������ࡱ���֡� ---------------------------------------

	//Mat tempTextTingJianGengDuo;

	//// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	//DUtil::RLSA_H(cannMat(rectTextTingJianGengDuo), tempTextTingJianGengDuo, 100);
	//DUtil::RLSA_V(tempTextTingJianGengDuo, tempTextTingJianGengDuo, 10);

	//vector<vector<Point>> countoursTextTingJianGengDuo = DUtil::getCountours(tempTextTingJianGengDuo);	// �����ͨͼ���������

	//for (auto countourTextTingJianGengDuo : countoursTextTingJianGengDuo)
	//{
	//	Rect tmpTextTingJianGengDuo = cv::boundingRect(countourTextTingJianGengDuo);	// ���Ҿ�������
	//	tmpTextTingJianGengDuo = tmpTextTingJianGengDuo + rectTextTingJianGengDuo.tl();			// �������ϽǶ�������
	//	textTingJianGengDuo.push_back(tmpTextTingJianGengDuo);

	//	cv::rectangle(mImg, tmpTextTingJianGengDuo, Scalar(0, 0, 255), 8);
	//}

	//// ����������ʼ��xλ�ö�Rect�����������
	//sort(textTingJianGengDuo.begin(), textTingJianGengDuo.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 19.���ײ�5��ͼ�꡿ ---------------------------------------

	//Mat tempBottonIcon;

	//// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	//DUtil::RLSA_H(cannMat(rectPicBottonIcon), tempBottonIcon, 20);
	//DUtil::RLSA_V(tempBottonIcon, tempBottonIcon, 20);

	//vector<vector<Point>> countoursBottonIcon = DUtil::getCountours(tempBottonIcon);	// �����ͨͼ���������

	//for (auto countourBottonIcon : countoursBottonIcon)
	//{
	//	Rect tmpPicBottonIcon = cv::boundingRect(countourBottonIcon);	// ���Ҿ�������
	//	tmpPicBottonIcon = tmpPicBottonIcon + rectPicBottonIcon.tl();			// �������ϽǶ�������
	//	picBottonIcon.push_back(tmpPicBottonIcon);

	//	cv::rectangle(mImg, tmpPicBottonIcon, Scalar(255, 0, 0), 8);
	//}

	//// ����������ʼ��xλ�ö�Rect�����������
	//sort(picBottonIcon.begin(), picBottonIcon.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });


	waitKey();
}

void main()
{
	//��ȡͼƬ
	Mat objImg = imread("750.jpg");		//�����Ӿ�ͼ
	cout << "�Ӿ�ͼ�Ŀ��볤��\t\t" << objImg.cols << " " << objImg.rows << endl;
	
	Mat srcImg = imread("375.jpg");	//����ͻ���ͼ��
	cout << "�ͻ���ͼ��Ŀ��볤��\t\t" << srcImg.cols << " " << srcImg.rows << endl;

	double rate = (float)objImg.cols / srcImg.cols;	//�Կͻ���ͼƬ������С
	Mat srcImg2 = srcImg.clone();
	resize(srcImg2, srcImg2, Size(), rate, rate);	//���տ�Ƚ��е�����������Ӿ�ͼһ������£��Ƚϱ߿���Ϣ��
	cout << "������ͻ���ͼ��Ŀ��볤��\t" << srcImg2.cols << " " << srcImg2.rows << endl;

	//ģ�ͼ�⣨�洢�Ӿ�ͼ�߿���Ϣ��
	vector<Rect> picSearch;				// ������
	vector<Rect> textTuiJian;			// ���ֿ��Ƽ�ſ�俴�㡱����
	Rect		 picTuiJian;			// �Ƽ�ͼƬ
	vector<Rect> picFourFunc;			// ����30�׵�4�ֹ���ͼƬ
	vector<Rect> textTuiJianGeDan;		// ���Ƽ��赥������
	vector<Rect> picTuiJianGeDan;		// ���Ƽ��赥��6��ͼƬ������
	vector<Rect> textXinTuiJian;		// �����Ƽ�������
	vector<Rect> picXinTuiJianLeft;		// �����Ƽ���ͼƬ+����������
	vector<Rect> picXinTuiJianRight;	// �����Ƽ���ͼƬ+����������
	vector<Rect> textMV;				// ��MV������
	vector<Rect> picMV;					// MVͼƬ+��������
	vector<Rect> textKanDian;			// �����㡱����
	vector<Rect> picKanDian;			// �����㡱ͼƬ
	vector<Rect> textYinYueRen;			// �������ˡ�����
	vector<Rect> picYinYueRen;			// ������ͼƬ
	vector<Rect> textPaJian;			// ��ſ�䡱����
	vector<Rect> picPaJian;				// ſ��ͼƬ
	vector<Rect> textTingJianGengDuo;	// ���������ࡱ����
	vector<Rect> picBottonIcon;			// �ײ�ͼ��

	// ���Ӿ�ͼƬ����λ�ü���ע
	DealModel(objImg, picSearch, textTuiJian, picTuiJian, picFourFunc, textTuiJianGeDan, picTuiJianGeDan, 
		textXinTuiJian, picXinTuiJianLeft, picXinTuiJianRight, textMV, picMV, textKanDian, picKanDian, 
		textYinYueRen, picYinYueRen, textPaJian, picPaJian, textTingJianGengDuo, picBottonIcon);

	for (auto rect_Client : picFourFunc)
	{
		cv::rectangle(objImg, rect_Client, Scalar(255, 0, 0), 8);
	}

	//ģ�ͼ�⣨�洢�ͻ���ͼ�߿���Ϣ��
	vector<Rect> picSearch_Client;			// 1.������
	vector<Rect> textTuiJian_Client;		// 2.���ֿ��Ƽ�ſ�俴�㡱����
	Rect		 picTuiJian_Client;			// 3.�Ƽ�ͼƬ
	vector<Rect> picFourFunc_Client;		// 4.����30�׵�4�ֹ���ͼƬ
	vector<Rect> textTuiJianGeDan_Client;	// 5.���Ƽ��赥������
	vector<Rect> picTuiJianGeDan_Client;	// 6.���Ƽ��赥��6��ͼƬ������
	vector<Rect> textXinTuiJian_Client;		// 7.�����Ƽ�������
	vector<Rect> picXinTuiJianLeft_Client;	// 8.�����Ƽ���ͼƬ+����������
	vector<Rect> picXinTuiJianRight_Client;	// 9.�����Ƽ���ͼƬ+����������
	vector<Rect> textMV_Client;				// 10.��MV������
	vector<Rect> picMV_Client;				// 11.MVͼƬ+��������
	vector<Rect> textKanDian_Client;		// 12.�����㡱����
	vector<Rect> picKanDian_Client;			// 13.�����㡱ͼƬ
	vector<Rect> textYinYueRen_Client;		// 14.�������ˡ�����
	vector<Rect> picYinYueRen_Client;		// 15.������ͼƬ
	vector<Rect> textPaJian_Client;			// 16.��ſ�䡱����
	vector<Rect> picPaJian_Client;			// 17.ſ��ͼƬ
	vector<Rect> textTingJianGengDuo_Client;// 18.���������ࡱ����
	vector<Rect> picBottonIcon_Client;		// 19.�ײ�ͼ��

	// �Կͻ���ͼƬ����λ�ü���ע
	DealModel(srcImg2, picSearch_Client, textTuiJian_Client, picTuiJian_Client, picFourFunc_Client, textTuiJianGeDan_Client, picTuiJianGeDan_Client,
		textXinTuiJian_Client, picXinTuiJianLeft_Client, picXinTuiJianRight_Client, textMV_Client, picMV_Client, textKanDian_Client, picKanDian_Client,
		textYinYueRen_Client, picYinYueRen_Client, textPaJian_Client, picPaJian_Client, textTingJianGengDuo_Client, picBottonIcon_Client);

	
	int flag = 1;	// �ж��Ƿ���Ҫ�ڿͻ���ͼ���л����Ӿ�ͼ��ܣ�1������0��������
	for (auto rect_ShiJue : picSearch)	// 1. ������ƥ��
	{
		flag = 0;
		for (auto rect_Client : picSearch_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textTuiJian)	// 2.���ֿ��Ƽ�ſ�俴�㡱����
	{
		flag = 0;
		for (auto rect_Client : textTuiJian_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	Rect rect_ShiJue = picTuiJian;	// 3.�Ƽ�ͼƬ
	Rect rect_Client = picTuiJian_Client;
	if (!((abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
		&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)))
	{
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picFourFunc)	// 4.����30�׵�4�ֹ���ͼƬ
	{
		flag = 0;
		for (auto rect_Client : picFourFunc_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textTuiJianGeDan)	// 5.���Ƽ��赥������
	{
		flag = 0;
		for (auto rect_Client : textTuiJianGeDan_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picTuiJianGeDan)	// 6.���Ƽ��赥��6��ͼƬ������
	{
		flag = 0;
		for (auto rect_Client : picTuiJianGeDan_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textXinTuiJian)	// 7.�����Ƽ�������
	{
		flag = 0;
		for (auto rect_Client : textXinTuiJian_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picXinTuiJianLeft)	// 8.�����Ƽ���ͼƬ+����������
	{
		flag = 0;
		for (auto rect_Client : picXinTuiJianLeft_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picXinTuiJianRight)	// 9.�����Ƽ���ͼƬ+����������
	{
		flag = 0;
		for (auto rect_Client : picXinTuiJianRight_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textMV)	// 10.��MV������
	{
		flag = 0;
		for (auto rect_Client : textMV_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picMV)	// 11.MVͼƬ+��������
	{
		flag = 0;
		for (auto rect_Client : picMV_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textKanDian)	// 12.�����㡱����
	{
		flag = 0;
		for (auto rect_Client : textKanDian_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picKanDian)	// 13.�����㡱ͼƬ
	{
		flag = 0;
		for (auto rect_Client : picKanDian_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textYinYueRen)	// 14.�������ˡ�����
	{
		flag = 0;
		for (auto rect_Client : textYinYueRen_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picYinYueRen)	// 15.������ͼƬ
	{
		flag = 0;
		for (auto rect_Client : picYinYueRen_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textPaJian)	// 16.��ſ�䡱����
	{
		flag = 0;
		for (auto rect_Client : textPaJian_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picPaJian)	// 17.ſ��ͼƬ
	{
		flag = 0;
		for (auto rect_Client : picPaJian_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	myshow(objImg, "�Ӿ�ͼ");		// ��ʾ��ƥ��ͼƬ
	myshow(srcImg2, "�ͻ���ͼ��");

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);  //ѡ��jpeg
	compression_params.push_back(100); //�����������Ҫ��ͼƬ����

	imwrite("ShiJue.jpg", objImg, compression_params);
	imwrite("Client.jpg", srcImg2, compression_params);

	waitKey();
}