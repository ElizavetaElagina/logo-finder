#include <windows.h>
#include <tchar.h> 
#include <stdio.h>
#include <strsafe.h>
#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//----------------DISTANCE BETWEEN POINTS--------
bool is_far(Point2f a, Point2f b) {
	double distance = sqrt((a.x - b.x) * (a.x - b.x) +
		(a.y - b.y) * (a.y - b.y));

	if (distance > 100)
		return true;
	return false;
}

//------------IS MATCH SUFFICIENT--------------
bool good_zone(vector<Point2f> corners) {
	int ind = 0;
	for (int i = 0; i < corners.size() - 2; i++)
		for (int j = i + 1; j < corners.size() - 1; j++)
			if (is_far(corners[i], corners[j]))
				for (int k = j + 1; k < corners.size(); k++)
					if (is_far(corners[i], corners[k]) && is_far(corners[j], corners[k]))
						return true;
	return false;
}

//-------------CALCULATING MATCH---------------
bool match(Mat logo, Mat scene) {
	int minHessian = 400;

	Ptr<SURF> detector = SURF::create(minHessian);
	std::vector<KeyPoint> keypoints_logo, keypoints_scene;

	detector->detect(logo, keypoints_logo);
	if (keypoints_logo.size() == 0)
		return false;
	detector->detect(scene, keypoints_scene);

	Ptr<SURF> extractor = SURF::create(minHessian);
	Mat descriptors_logo, descriptors_scene;

	extractor->compute(logo, keypoints_logo, descriptors_logo);
	extractor->compute(scene, keypoints_scene, descriptors_scene);

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_logo, descriptors_scene, matches);

	double max_dist = 0; double min_dist = 100;

	for (int i = 0; i < descriptors_logo.rows; i++) {
		double dist = matches[i].distance;		
		if (dist < min_dist)
			min_dist = dist;	
		if (dist > max_dist)
			max_dist = dist; 
	}

	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_logo.rows; i++) 
		if (matches[i].distance < 3 * min_dist)
			good_matches.push_back(matches[i]);
		
	Mat img_matches;
	drawMatches(logo, keypoints_logo, scene, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	std::vector<Point2f> obj;
	std::vector<Point2f> picture;

	for (int i = 0; i < good_matches.size(); i++) {	
		obj.push_back(keypoints_logo[good_matches[i].queryIdx].pt);	
		picture.push_back(keypoints_scene[good_matches[i].trainIdx].pt); 
	}

	Mat H = findHomography(obj, picture, CV_RANSAC);
	if (H.dims == 0)
		return false;
	
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(logo.cols, 0);
	obj_corners[2] = cvPoint(logo.cols, logo.rows); obj_corners[3] = cvPoint(0, logo.rows);

	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);

	return good_zone(scene_corners);
}

//==========MAIN=========
int main(int argc, char** argv){
	WIN32_FIND_DATA ffd;
	LARGE_INTEGER filesize;
	TCHAR szDir[MAX_PATH];
	size_t length_of_arg;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	DWORD dwError = 0;
	Mat img_scene, img_object;
	
	if (argc > 1) {  
		img_scene = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
		if (argc > 2 && strlen(argv[2]) != 1) { // logo is given
			img_object = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
			if (match(img_object, img_scene))
				cout << "Logo found\n";
			else
				cout << "Logo not found";
			return 1;
		}
	}
	
	else
	    img_scene = imread("DataSet_pic/the_rolling_stones_by_evie128-d6cjfax.png", CV_LOAD_IMAGE_GRAYSCALE);
	
	const wchar_t path[25] = L"DataSet_pic/reference";
	StringCchLength(path, MAX_PATH, &length_of_arg);
	StringCchCopy(szDir, MAX_PATH, path);
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));

	hFind = FindFirstFile(szDir, &ffd);

	if (INVALID_HANDLE_VALUE != hFind)
	    do	{
		    if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)){
			    filesize.LowPart = ffd.nFileSizeLow;
			    filesize.HighPart = ffd.nFileSizeHigh;
			
		     	wstring ws(ffd.cFileName);
		    	string str(ws.begin(), ws.end());
		    	string s = "DataSet_pic\\reference\\" + str;
		
			    img_object = imread(s, CV_LOAD_IMAGE_GRAYSCALE);
		
				if (!img_object.data || !img_scene.data) {
					std::cout << " --(!) Error reading image " << std::endl;
					return -1;
				}
			
				if (match(img_object, img_scene)) {
				    cout << "Logo found: " << str << endl;
				    if (argc > 2) // no searching for another match
					    return 1;
			     }
			     else
				    cout << "Logo not found: " << str << endl;
		      }
    	} 
	    while (FindNextFile(hFind, &ffd) != 0);
 
	FindClose(hFind);
	system("PAUSE");
	return dwError;
}


