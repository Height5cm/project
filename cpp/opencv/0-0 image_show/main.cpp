#include<opencv2\opencv.hpp>
using namespace cv;
int main()
{
    Mat img = imread("grey.png");
	if (img.empty())
		return 0;
	namedWindow("window1",WINDOW_AUTOSIZE);
	imshow("window1", img);
    waitKey(5000);
	destroyWindow("window1");
}