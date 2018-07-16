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

//模型检测：分别对图像上中下三块区域进行检测
void DealModel(const Mat &src, vector<Rect> &targ_search, vector<Rect> &targ_up_TuiJian, Rect &targ_up,
	vector<Rect> &rects_middle_targ, vector<Rect> &rects_middle_targ_text,
	vector<Rect> &rects_down_targ, vector<Rect> &rects_down_targ_text, vector<Rect> &rects_down_targ_two)
{
	Mat mImg = src;

	//将带匹配图片分别划分为rect_up、rect_middle、rect_down三个区域，分别用Rect类型变量保存
	//0,450|1079,780 大图1的大致区域（通过左上角及右下角两点确定矩形）
	//0,840|1079,1010
	//0,1240|1079,2060

	/*cout << mImg.cols << endl;
	cout << mImg.rows << endl;*/

	const Rect rect_up_Search(Point(0, 110), Point(mImg.cols, 250));	//最上边搜索条
	const Rect rect_up_TuiJian(Point(100, 290), Point(700, 392));		//最上边文字“乐库”等
	const Rect rect_up(Point(0, 450), Point(mImg.cols, 780));			//最上边大图
	const Rect rect_middle(Point(0, 840), Point(mImg.cols, 1010));		//中间图标区域
	const Rect rect_middle_BiTing(Point(0, 1110), Point(mImg.cols, 1190));//中间文字“必听歌单”
																		  //const Rect rect_down(Point(0, 1240), Point(mImg.cols, 2070));		//下边6张专辑图片
	const Rect rect_down(Point(0, 1240), Point(mImg.cols, 1540));		//下边3张专辑图片
	const Rect rect_botton_GeQuName(Point(0, 1570), Point(mImg.cols, 1700));//下面第一排3张图片的歌曲名称
	const Rect rect_down_two(Point(0, 1760), Point(mImg.cols, 2070));	//最下边3张专辑图片

	Mat cannMat;
	//cv::Canny(mImg, cannMat, 50, 150);
	cv::Canny(mImg, cannMat, 10, 50);					//对输入的待检测图片做边缘检测

														//--------------------------------------- 最上方搜索条定位 ---------------------------------------

	Mat tempUpSearch;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rect_up_Search), tempUpSearch, 5);			// 连接水平像素距离小于35的点
	DUtil::RLSA_V(tempUpSearch, tempUpSearch, 5);						// 连接垂直像素距离小于35的点
																		// 获得连通图轮廓点矩阵（轮廓应该有4个，所有边缘点位于这4个轮廓上）
	vector<vector<Point>> countours_up_search = DUtil::getCountours(tempUpSearch);

	for (auto countour_up_search : countours_up_search)
	{
		// 矩形轮廓查找
		Rect rect_up_search = cv::boundingRect(countour_up_search);
		rect_up_search = rect_up_search + rect_up_Search.tl();			// rect_middle.tl返回原中间大区域左上角顶点坐标
		if (rect_up_search.x > 170 && rect_up_search.x < 900)
			continue;
		targ_search.push_back(rect_up_search);		// rects_middle_targ保存检测出轮廓的准确的矩形数组信息（其中应该包含4个Rect信息）

													// 显示最上面时间信号矩形轮廓
		cv::rectangle(mImg, rect_up_search, Scalar(0, 255, 0), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序（最终结果为rects_middle_targ中依次存储中间从左到右的4个Rect信息）
	sort(targ_search.begin(), targ_search.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 最上方文字定位 ---------------------------------------

	Mat tempUpText;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rect_up_TuiJian), tempUpText, 10);			// 连接水平像素距离小于35的点
	DUtil::RLSA_V(tempUpText, tempUpText, 10);						// 连接垂直像素距离小于35的点
																	// 获得连通图轮廓点矩阵（轮廓应该有4个，所有边缘点位于这4个轮廓上）
	vector<vector<Point>> countours_up_text = DUtil::getCountours(tempUpText);

	for (auto countour_up_text : countours_up_text)
	{
		// 矩形轮廓查找
		Rect rect_up_text = cv::boundingRect(countour_up_text);
		rect_up_text = rect_up_text + rect_up_TuiJian.tl();			// rect_middle.tl返回原中间大区域左上角顶点坐标
		targ_up_TuiJian.push_back(rect_up_text);		// rects_middle_targ保存检测出轮廓的准确的矩形数组信息（其中应该包含4个Rect信息）

														// 显示最上面文字矩形轮廓
		cv::rectangle(mImg, rect_up_text, Scalar(0, 255, 0), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序（最终结果为rects_middle_targ中依次存储中间从左到右的4个Rect信息）
	sort(targ_up_TuiJian.begin(), targ_up_TuiJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 最上方大图定位 ---------------------------------------

	//水平垂直投影
	vector<double>  hproj = DUtil::horizontalProjection(cannMat(rect_up));	//水平投影数组
	vector<double>  vproj = DUtil::verticalProjection(cannMat(rect_up));	//垂直投影数组

																			//查找边界
	vector<int> hds = DUtil::findIndex(hproj, 15);	//竖线中超过15个像素点开始作为图像区域
	vector<int> vds = DUtil::findIndex(vproj, 15);	//横线中超过15个像素点开始作为图像区域

	Point pt1 = Point(hds[0], vds[0]);
	Point pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);
	targ_up = Rect(pt1, pt2);						//通过投影像素点超过临界值，确定准确图像范围

													//偏移（将图像相对原定义Rect粗略矩形位置进行位移）最终targ_up中存储准确识别出的上方大图的Rect信息
	targ_up.x += rect_up.x;
	targ_up.y += rect_up.y;

	//--------------------------------------- 中间4张小图区域定位 ---------------------------------------

	Mat tempMiddle;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rect_middle), tempMiddle, 35);			// 连接水平像素距离小于35的点
	DUtil::RLSA_V(tempMiddle, tempMiddle, 35);						// 连接垂直像素距离小于35的点

																	// 获得连通图轮廓点矩阵（轮廓应该有4个，所有边缘点位于这4个轮廓上）
	vector<vector<Point>> countours = DUtil::getCountours(tempMiddle);
	//	vector<Rect> rects_middle_targ;

	for (auto countour : countours)
	{
		//矩形轮廓查找
		Rect rect = cv::boundingRect(countour);
		rect = rect + rect_middle.tl();			// rect_middle.tl返回原中间大区域左上角顶点坐标
		rects_middle_targ.push_back(rect);		// rects_middle_targ保存检测出轮廓的准确的矩形数组信息（其中应该包含4个Rect信息）
	}

	// 按照轮廓开始的x位置对Rect数组进行排序（最终结果为rects_middle_targ中依次存储中间从左到右的4个Rect信息）
	sort(rects_middle_targ.begin(), rects_middle_targ.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 中间“必听歌单”文字定位 ---------------------------------------

	Mat tempMiddleText;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rect_middle_BiTing), tempMiddleText, 40);			// 连接水平像素距离小于35的点
	DUtil::RLSA_V(tempMiddleText, tempMiddleText, 40);						// 连接垂直像素距离小于35的点
																			// 获得连通图轮廓点矩阵（轮廓应该有4个，所有边缘点位于这4个轮廓上）
	vector<vector<Point>> countours_middle_text = DUtil::getCountours(tempMiddleText);

	for (auto countour_middle_text : countours_middle_text)
	{
		// 矩形轮廓查找
		Rect rect_middle_text = cv::boundingRect(countour_middle_text);
		rect_middle_text = rect_middle_text + rect_middle_BiTing.tl();			// rect_middle.tl返回原中间大区域左上角顶点坐标
		rects_middle_targ_text.push_back(rect_middle_text);		// rects_middle_targ保存检测出轮廓的准确的矩形数组信息（其中应该包含4个Rect信息）

																// 显示中间文字矩形轮廓
		cv::rectangle(mImg, rect_middle_text, Scalar(0, 255, 0), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序（最终结果为rects_middle_targ中依次存储中间从左到右的4个Rect信息）
	sort(rects_middle_targ_text.begin(), rects_middle_targ_text.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 下面3张图片区域定位 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rect_down));
	vproj = DUtil::verticalProjection(cannMat(rect_down));

	//查找边界
	double th = rect_down.width*0.1;
	vds = DUtil::findIndex(vproj, th);		// 将像素点超过宽度*0.1的边进行标记，查找准确矩形范围
	hds = DUtil::findIndex(hproj, th);
	pt1 = Point(hds[0], vds[0]);
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);
	Rect rect_down_targ = Rect(pt1, pt2);

	//偏移
	rect_down_targ.y += rect_down.y;

	//水平和垂直投影
	DUtil::RLSA_H(cannMat(rect_down_targ), tempMiddle, 5);	// 将像素距离小于5的像素点进行连接
	DUtil::RLSA_V(tempMiddle, tempMiddle, 5);

	//轮廓查找（应该有6个轮廓）
	countours = DUtil::getCountours(tempMiddle);
	//	vector<Rect> &rects_down_targ

	vector<Rect> tempRect1;								// 分别保存6张两排图像的Rect信息（每个vector中保存3张图片信息）

	for (int i = 0; i < 3; i++)
	{
		//矩形轮廓查找
		Rect rect = cv::boundingRect(countours[i]);
		rect = rect + rect_down_targ.tl();			// 根据矩形大区域左上角顶点坐标进行平移
		tempRect1.push_back(rect);				// 下面一排图像信息存储在tempRect1数组中，
	}

	//从左到右排序
	sort(tempRect1.begin(), tempRect1.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//记录数据（按照从上到下，从左到右的顺序将6张图片信息依次存储进rects_down_targ数组中）
	for (int i = 0; i < tempRect1.size(); i++)
	{
		rects_down_targ.push_back(tempRect1[i]);
	}

	//--------------------------------------- 下面第一排3张图片文字定位 ---------------------------------------

	Mat tempBottonText;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rect_botton_GeQuName), tempBottonText, 30);			// 连接水平像素距离小于35的点
	DUtil::RLSA_V(tempBottonText, tempBottonText, 30);						// 连接垂直像素距离小于35的点
																			// 获得连通图轮廓点矩阵（轮廓应该有4个，所有边缘点位于这4个轮廓上）
	vector<vector<Point>> countours_botton_text = DUtil::getCountours(tempBottonText);

	for (auto countour_botton_text : countours_botton_text)
	{
		// 矩形轮廓查找
		Rect rect_botton_text = cv::boundingRect(countour_botton_text);
		rect_botton_text = rect_botton_text + rect_botton_GeQuName.tl();			// rect_middle.tl返回原中间大区域左上角顶点坐标
		rects_down_targ_text.push_back(rect_botton_text);		// rects_middle_targ保存检测出轮廓的准确的矩形数组信息（其中应该包含4个Rect信息）

																// 显示中间文字矩形轮廓
		cv::rectangle(mImg, rect_botton_text, Scalar(0, 255, 0), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序（最终结果为rects_middle_targ中依次存储中间从左到右的4个Rect信息）
	sort(rects_down_targ_text.begin(), rects_down_targ_text.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 最下面3张图片区域定位 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rect_down_two));
	vproj = DUtil::verticalProjection(cannMat(rect_down_two));

	//查找边界
	double th2 = rect_down_two.width*0.1;
	vds = DUtil::findIndex(vproj, th2);		// 将像素点超过宽度*0.1的边进行标记，查找准确矩形范围
	hds = DUtil::findIndex(hproj, th2);
	pt1 = Point(hds[0], vds[0]);
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);
	Rect rect_down_targ_two = Rect(pt1, pt2);

	//偏移
	rect_down_targ_two.y += rect_down_two.y;

	//水平和垂直投影
	DUtil::RLSA_H(cannMat(rect_down_targ_two), tempMiddle, 5);	// 将像素距离小于5的像素点进行连接
	DUtil::RLSA_V(tempMiddle, tempMiddle, 5);

	//轮廓查找（应该有6个轮廓）
	countours = DUtil::getCountours(tempMiddle);
	//	vector<Rect> &rects_down_targ

	vector<Rect> tempRect2;								// 分别保存6张两排图像的Rect信息（每个vector中保存3张图片信息）

	for (int i = 0; i < 3; i++)
	{
		//矩形轮廓查找
		Rect rect2 = cv::boundingRect(countours[i]);
		rect2 = rect2 + rect_down_targ_two.tl();			// 根据矩形大区域左上角顶点坐标进行平移
		tempRect2.push_back(rect2);				// 下面一排图像信息存储在tempRect1数组中
	}

	//从左到右排序
	sort(tempRect2.begin(), tempRect2.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//记录数据（按照从上到下，从左到右的顺序将6张图片信息依次存储进rects_down_targ数组中）
	for (int i = 0; i < tempRect2.size(); i++)
	{
		rects_down_targ_two.push_back(tempRect2[i]);
	}
}

//检测图像定位
void DealDetectImg(const Mat &src, vector<Rect> &targ_up_search_detect, vector<Rect> &targ_up_text, Rect &targ_up,
	vector<Rect> &rects_middle, vector<Rect> &rects_middle_text,
	vector<Rect> &down_rects, vector<Rect> &down_rects_text, vector<Rect> &down_rects_two)
{
	Mat imgsrc = src;
	Mat thMat;

	Mat mImg = src;

	//白色过滤（去除白色背景）
	inRange(imgsrc, Scalar(240, 240, 240), Scalar(255, 255, 255), thMat);
	thMat = ~thMat;

	//整张图片（检测范围）的右下点坐标
	Point ptRightDown(imgsrc.cols - 1, 0.92*imgsrc.rows);	// 不包括最下方个性化菜单图标检测
	Rect targRect(Point(0, 0), ptRightDown);

	Mat cannMat;
	DUtil::GaussianBlur(imgsrc, imgsrc, 5);	//高斯模糊
	cv::Canny(imgsrc, cannMat, 20, 30);		//边缘检测

	Mat cannMatClose = DUtil::ImgClose(cannMat, 10);	//闭运算
	DUtil::RLSA_H(cannMat, cannMatClose, 10);			//对水平和垂直方向像素距离小于10的像素点进行连接
	DUtil::RLSA_V(cannMatClose, cannMatClose, 10);

	// 视觉图坐标位置信息
	const Rect rect_up_Search(Point(0, 100), Point(imgsrc.cols, 0.1*imgsrc.rows));	//最上边搜索条
	const Rect rect_up_TuiJian(Point(0.064*imgsrc.cols, 0.13*imgsrc.rows), Point(0.67*imgsrc.cols, 0.19*imgsrc.rows));		//最上边文字“乐库”等
																															//const Rect rect_up(Point(0, 450), Point(mImg.cols, 780));			//最上边大图
																															//const Rect rect_middle(Point(0, 840), Point(mImg.cols, 1010));		//中间图标区域
	const Rect rect_middle_BiTing(Point(0.055*imgsrc.cols, 0.58*imgsrc.rows), Point(0.95*imgsrc.cols, 0.62*imgsrc.rows));//中间文字“必听歌单”
																														 //																	  //const Rect rect_down(Point(0, 1240), Point(mImg.cols, 2070));		//下边6张专辑图片
																														 //const Rect rect_down(Point(0, 1240), Point(mImg.cols, 1540));		//下边3张专辑图片
	const Rect rect_botton_GeQuName(Point(51, 0.818*imgsrc.rows), Point(0.9*imgsrc.cols, 0.87*imgsrc.rows));//下面第一排3张图片的歌曲名称
	cv::rectangle(mImg, rect_botton_GeQuName, Scalar(0, 0, 255), 8);
	//const Rect rect_down_two(Point(0, 1760), Point(mImg.cols, 2070));	//最下边3张专辑图片

	//------------------------------------ 获取最上面搜索条位置信息 ------------------------------------

	Mat tempUpSearch;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rect_up_Search), tempUpSearch, 5);			// 连接水平像素距离小于35的点
	DUtil::RLSA_V(tempUpSearch, tempUpSearch, 5);						// 连接垂直像素距离小于35的点
																		// 获得连通图轮廓点矩阵（轮廓应该有4个，所有边缘点位于这4个轮廓上）
	vector<vector<Point>> countours_up_search = DUtil::getCountours(tempUpSearch);

	for (auto countour_up_search : countours_up_search)
	{
		// 矩形轮廓查找
		Rect rect_up_search = cv::boundingRect(countour_up_search);
		rect_up_search = rect_up_search + rect_up_Search.tl();			// rect_middle.tl返回原中间大区域左上角顶点坐标
		if (rect_up_search.x > 170 && rect_up_search.x < 900)
			continue;
		targ_up_search_detect.push_back(rect_up_search);		// rects_middle_targ保存检测出轮廓的准确的矩形数组信息（其中应该包含4个Rect信息）

		cv::rectangle(mImg, rect_up_search, Scalar(0, 255, 0), 8);	// 显示最上面时间信号矩形轮廓
	}

	// 按照轮廓开始的x位置对Rect数组进行排序（最终结果为rects_middle_targ中依次存储中间从左到右的4个Rect信息）
	sort(targ_up_search_detect.begin(), targ_up_search_detect.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 最上方“推荐”文字定位 ---------------------------------------

	Mat tempUpText;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rect_up_TuiJian), tempUpText, 10);			// 连接水平像素距离小于35的点
	DUtil::RLSA_V(tempUpText, tempUpText, 10);						// 连接垂直像素距离小于35的点
																	// 获得连通图轮廓点矩阵（轮廓应该有4个，所有边缘点位于这4个轮廓上）
	vector<vector<Point>> countours_up_text = DUtil::getCountours(tempUpText);

	for (auto countour_up_text : countours_up_text)
	{
		// 矩形轮廓查找
		Rect rect_up_text = cv::boundingRect(countour_up_text);
		rect_up_text = rect_up_text + rect_up_TuiJian.tl();			// rect_middle.tl返回原中间大区域左上角顶点坐标
		targ_up_text.push_back(rect_up_text);		// rects_middle_targ保存检测出轮廓的准确的矩形数组信息（其中应该包含4个Rect信息）

		cv::rectangle(mImg, rect_up_text, Scalar(0, 255, 0), 8);	// 显示最上面文字矩形轮廓
	}

	// 按照轮廓开始的x位置对Rect数组进行排序（最终结果为rects_middle_targ中依次存储中间从左到右的4个Rect信息）
	sort(targ_up_text.begin(), targ_up_text.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//------------------------------------ 获取最上面图片位置信息 ------------------------------------
	vector<vector<Point>> countours = DUtil::getCountours(cannMat(targRect));
	targ_up = cv::boundingRect(countours[0]);	//找到最上面面积最大的大图

	cv::rectangle(mImg, targ_up, Scalar(0, 255, 0), 8);

	//------------------------------------ 获取中间4个小图标位置信息 ------------------------------------
	//查找中间区域

	Rect rect_middle_temp = targ_up;	//中间区域
	rect_middle_temp.y = rect_middle_temp.br().y*1.05;		// y轴起始坐标从上面图片下边0.05位置开始检测
	rect_middle_temp.height *= 0.8;							// 中间矩形y值大概为上面矩形y值的0.8倍

	Mat tempMiddle;

	//水平链接
	DUtil::RLSA_H(cannMat(rect_middle_temp), tempMiddle, 35);
	DUtil::RLSA_V(tempMiddle, tempMiddle, 35);

	countours = DUtil::getCountours(tempMiddle);
	//	vector<Rect> rects_middle;

	for (auto countour : countours)
	{
		//矩形轮廓查找
		Rect rectTemp = cv::boundingRect(countour);
		rectTemp = rectTemp + rect_middle_temp.tl();
		if (rectTemp.width < 100)
			continue;
		rects_middle.push_back(rectTemp);
		cv::rectangle(imgsrc, rectTemp, Scalar(255, 0, 255), 8);
	}
	//从左到右排序
	sort(rects_middle.begin(), rects_middle.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 中间“必听歌单”文字定位 ---------------------------------------

	Mat tempMiddleText;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rect_middle_BiTing), tempMiddleText, 40);			// 连接水平像素距离小于35的点
	DUtil::RLSA_V(tempMiddleText, tempMiddleText, 40);						// 连接垂直像素距离小于35的点
																			// 获得连通图轮廓点矩阵（轮廓应该有4个，所有边缘点位于这4个轮廓上）
	vector<vector<Point>> countours_middle_text = DUtil::getCountours(tempMiddleText);

	for (auto countour_middle_text : countours_middle_text)
	{
		// 矩形轮廓查找
		Rect rect_middle_text = cv::boundingRect(countour_middle_text);
		rect_middle_text = rect_middle_text + rect_middle_BiTing.tl();			// rect_middle.tl返回原中间大区域左上角顶点坐标
		rects_middle_text.push_back(rect_middle_text);		// rects_middle_targ保存检测出轮廓的准确的矩形数组信息（其中应该包含4个Rect信息）

		cv::rectangle(mImg, rect_middle_text, Scalar(0, 255, 0), 8);	// 显示中间文字矩形轮廓
	}

	// 按照轮廓开始的x位置对Rect数组进行排序（最终结果为rects_middle_targ中依次存储中间从左到右的4个Rect信息）
	sort(rects_middle_text.begin(), rects_middle_text.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 下面第一排3张图片文字定位 ---------------------------------------

	Mat tempBottonText;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rect_botton_GeQuName), tempBottonText, 30);			// 连接水平像素距离小于35的点
	DUtil::RLSA_V(tempBottonText, tempBottonText, 30);						// 连接垂直像素距离小于35的点
																			// 获得连通图轮廓点矩阵（轮廓应该有4个，所有边缘点位于这4个轮廓上）
	vector<vector<Point>> countours_botton_text = DUtil::getCountours(tempBottonText);

	for (auto countour_botton_text : countours_botton_text)
	{
		// 矩形轮廓查找
		Rect rect_botton_text = cv::boundingRect(countour_botton_text);
		rect_botton_text = rect_botton_text + rect_botton_GeQuName.tl();			// rect_middle.tl返回原中间大区域左上角顶点坐标
		down_rects_text.push_back(rect_botton_text);		// rects_middle_targ保存检测出轮廓的准确的矩形数组信息（其中应该包含4个Rect信息）

		cv::rectangle(mImg, rect_botton_text, Scalar(0, 255, 0), 8);	// 显示中间文字矩形轮廓
	}

	// 按照轮廓开始的x位置对Rect数组进行排序（最终结果为rects_middle_targ中依次存储中间从左到右的4个Rect信息）
	sort(down_rects_text.begin(), down_rects_text.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//------------------------------------ 获取下面6张图片位置信息 ------------------------------------
	//Rect rect_down = Rect(Point(0, rect_middle_temp.y + rect_middle_temp.height*1.4),ptRightDown);
	Rect rect_down = Rect(Point(0, 0.65*imgsrc.rows), ptRightDown);
	Mat tempMat = thMat(rect_down) | cannMatClose(rect_down);

	countours = DUtil::getCountours(tempMat);
	float width_rect = 0;
	float height_rect = 0;
	int miny = 30000, maxy = 0;

	for (int j = 0; j < 3; j++)
	{
		Rect rect0 = cv::boundingRect(countours[j]);	//最大区域的大图
		rect0.y += rect_down.y;
		width_rect += rect0.width;		//宽度累加
		height_rect += rect0.height;
		down_rects.push_back(rect0);

		miny = min(miny, rect0.y);		//最大值
		maxy = max(maxy, rect0.y);
	}
	/********************************************/
	width_rect /= 3;
	height_rect /= 3;
	for (int j = 3; j < countours.size(); j++)
	{
		Rect rect0 = cv::boundingRect(countours[j]);//最大区域的大图
		if (down_rects.size() == 6)//6个就退出
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
	//3个（只检测到一半图片）
	if (down_rects.size() == 3)
	{
		sort(down_rects.begin(), down_rects.end(), [](const Rect &r1, const Rect &r2)	//只对上面3张图片位置进行检测
		{return r1.x < r2.x; });
	}
	else if (down_rects.size() == 6)
	{
		int thy = (miny + maxy) / 2;
		vector<Rect> tempRect1, tempRect2;
		for (int j = 0; j < 6; j++)
		{
			//矩形轮廓查找
			Rect rect = down_rects[j];
			if (rect.y > thy)//上下分离
			{
				tempRect1.push_back(rect);
			}
			else
				tempRect2.push_back(rect);
		}

		//从左到右排序（先排第二排，再排第一排）
		sort(tempRect1.begin(), tempRect1.end(), [](const Rect &r1, const Rect &r2)
		{return r1.x < r2.x; });

		sort(tempRect2.begin(), tempRect2.end(), [](const Rect &r1, const Rect &r2)
		{return r1.x < r2.x; });

		down_rects.clear();

		//记录数据
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

// 计算匹配度
float MatchRate(Rect r1, Rect r2)
{
	Rect rr0 = r1 & r2;			// 对两区域取交集
	Rect rr1 = r2 | r1;			// 对两区域取并集
	return  rr0.area() / (float)rr1.area();		// 定义匹配度=图像重叠面积/总和面积
}


void main()
{
	//读取图片
	Mat mImg = imread("1.jpg");				//读入输入待匹配图像
	Size msize = mImg.size();				//计算带匹配图像大小，作为模板
	Mat libImgSrc = imread("2.jpg");		//读入视觉库图片

											//视觉图中每幅图片宽度及高度定义（以模板作为依据）
	int width = 450;
	int height = 799;

	//将视觉图手动划分为5张图片，按照像素划分区域
	Rect rect1(0, 0, width, height);
	Rect rect2(450, 0, width, height);
	Rect rect3(900, 0, width, height);
	Rect rect4(1350, 0, width, height);
	Rect rect5(1800, 0, 674, 1199);

	//Rect类型数组rects中依次保存视觉库中的5张图片
	vector<Rect> rects;
	rects.push_back(rect1);
	rects.push_back(rect2);
	rects.push_back(rect3);
	rects.push_back(rect4);
	rects.push_back(rect5);

	//模型检测（Rect类变量/数组targ_up、rects_middle_targ、rects_down_targ分别保存图片上中下三区域信息）
	vector<Rect> targ_top_time;				// 最上方时间信号信息显示
	vector<Rect> targ_search;				// 上方“搜索”横条位置
	vector<Rect> targ_up_TuiJian;			// 检测最上方“推荐乐库等文字”
	Rect		 targ_up;					// 最上方大图
	vector<Rect> rects_middle_targ;			// 中间4张小图
	vector<Rect> rects_middle_targ_text;	// 检测中间“必听歌单”文字
	vector<Rect> rects_down_targ;			// 下面第一排3张图片
	vector<Rect> rects_down_targ_two;		// 下面第二排3张图片
	vector<Rect> rects_down_targ_text;		// 检测最下方图片下文字

											//------------------ 模型检测（分别对待匹配图片上中下三块区域进行检测）-------------------
	DealModel(mImg, targ_search, targ_up_TuiJian, targ_up, rects_middle_targ, rects_middle_targ_text, rects_down_targ, rects_down_targ_text, rects_down_targ_two);

	//--------------------------------- 显示原待检测图片边框 ---------------------------------
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
	myshow(mImg, "mImg");		// 显示待匹配图片（标记边框）

								//--------------------------------- 找到视觉库中图片边框，计算匹配度 ---------------------------------
	float max_match_rate = 0;	// 保存最大匹配图

	int index = 0;							// 保存视觉库中最匹配图片的序号、上中下边框位置信息
	Rect		 targ_up_detect_index;		//上
	vector<Rect> rects_middle_detect_index; //中
	vector<Rect> down_rects_detect_index;   //下
	Mat img_index;


	for (int i = 0; i < 3; i++)
	{
		cout << i << endl;
		Mat img2 = libImgSrc(rects[i]).clone();			//按照序号依次复制视觉库图片

		double rate = (float)msize.width / img2.cols;	//对视觉库中的图片调整大小
		resize(img2, img2, Size(), rate, rate);			//按照宽度进行调整（宽度与带匹配图片一致情况下，比较边框信息）

		vector<Rect> targ_up_time_detect;			// 最上面时间信号图片
		vector<Rect> targ_up_search_detect;			// 最上面搜索框
		vector<Rect> targ_up_text_detect;			// 最上面“推荐”文字
		Rect		 targ_up_detect;				// 最上面大图
		vector<Rect> rects_middle_detect;			// 中间4张小图
		vector<Rect> rects_middle_text_detect;		// 中间“必听歌单”文字
		vector<Rect> down_rects_detect;				// 下面第一排3张图片
		vector<Rect> down_rects_text_detect;		// 下面第一排3张图片名字信息
		vector<Rect> down_rects_detect_two;			// 下面第二排3张图片

		DealDetectImg(img2, targ_up_search_detect, targ_up_text_detect, targ_up_detect, rects_middle_detect, rects_middle_text_detect, down_rects_detect, down_rects_text_detect, down_rects_detect_two);	//对视觉库中图片依次获取边框信息

																																																			//----------------------------------------- 计算匹配度与显示结果 -----------------------------------------
		float match_up_time = 0, match_up_search = 0, match_up_text = 0, match_middle_text = 0, match_down_text = 0;

		//---------------------------------- 对搜索框位置进行匹配 ----------------------------------
		for (int j = 0; j<3; j++)
		{
			match_up_search += MatchRate(targ_search[j], targ_up_search_detect[j]);
		}
		match_up_search = match_up_search / 4;
		cout << "搜索框位置匹配度：" << match_up_search << endl;

		//---------------------------------- 对“推荐”文字位置进行匹配 ----------------------------------
		for (int j = 0; j<4; j++)
		{
			match_up_text += MatchRate(targ_up_TuiJian[j], targ_up_text_detect[j]);
		}
		match_up_text = match_up_text / 4;
		cout << "标题文字匹配度：" << match_up_text << endl;

		//---------------------------------- 对上方图片进行匹配 -----------------------------------------

		float match_up = MatchRate(targ_up, targ_up_detect);
		cout << "最上面大图匹配度：" << match_up << endl;

		//---------------------------------- 对中间4张图标区域进行匹配 ----------------------------------
		float match_middle = 0;
		for (int j = 0; j<4; j++)
		{
			match_middle += MatchRate(rects_middle_targ[j], rects_middle_detect[j]);
		}
		match_middle = match_middle / 4;
		cout << "中间4张每日歌曲等位置匹配：" << match_middle << endl;

		//---------------------------------- 对“必听歌曲”与“更多”文字位置进行匹配 ----------------------------------
		for (int j = 0; j<2; j++)
		{
			match_middle_text += MatchRate(rects_middle_targ_text[j], rects_middle_text_detect[j]);
		}
		match_middle_text = match_middle_text / 2;
		cout << "必听歌曲与更多文字位置匹配度：" << match_middle_text << endl;

		//---------------------------------- 对下面6张图标区域进行匹配 ----------------------------------
		float match_down = 0;
		for (int j = 0; j < down_rects_detect.size(); j++)
		{
			float r = MatchRate(down_rects_detect[j], rects_down_targ[j]);
			//			if(r>0.2)
			match_down += r;
		}
		match_down = match_down / down_rects_detect.size();
		cout << "下匹配：" << match_down << endl;

		// 计算平均匹配度，打印输出
		float match_rate = (match_down + match_middle + match_up) / 3;
		cout << "平均匹配:" << match_rate << endl;


		//if (match_rate > max_match_rate)		// 找到匹配度最大的视觉图，记录序号与边框位置信息
		//{
		//	max_match_rate = match_rate;
		//	index = i;

		//	targ_up_detect_index = targ_up_detect;			//上
		//	rects_middle_detect_index = rects_middle_detect;//中
		//	down_rects_detect_index = down_rects_detect;	//下
		//	img_index = img2.clone();
		//}

		//--------------------------------- 显示检测图片结果 ---------------------------------

		//// 依次显示视觉图图片，同时画出自己的边框
		//cv::rectangle(img2, targ_up_detect, Scalar(0, 0, 255), 8);
		//for (auto rect : down_rects_detect)
		//{
		//	cv::rectangle(img2, rect, Scalar(0, 0, 255), 8);
		//}
		//for (auto rect : rects_middle_detect)
		//{
		//	cv::rectangle(img2, rect, Scalar(0, 0, 255), 8);
		//}

		////依次显示视觉图中的图片，同时画出原待匹配图形的边框
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

	////在最大匹配度的视觉图中显示自己的边框
	//for (auto rect : down_rects_detect_index)
	//{
	//	cv::rectangle(img_index, rect, Scalar(0, 0, 255), 8);
	//}
	//for (auto rect : rects_middle_detect_index)
	//{
	//	cv::rectangle(img_index, rect, Scalar(0, 0, 255), 8);
	//}
	//cv::rectangle(img_index, targ_up_detect_index, Scalar(0, 0, 255), 8);

	////在最大匹配度的视觉图中显示原来待匹配图片的边框
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