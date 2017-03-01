#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
//#include <Windows.h>
#include "headers.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <omp.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>

typedef dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > > frontal_face_detector;
using namespace cv;
using namespace std;

void DrawPredictedImage(cv::Mat_<uchar> image, cv::Mat_<double>& shape){
	for (int i = 0; i < shape.rows; i++){
		cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, (255));
	}
	cv::imshow("show image", image);
	cv::waitKey(0);
}

void Test(const char* config_file_path){
	cout << "parsing config_file: " << config_file_path << endl;

    ifstream fin;
    fin.open(config_file_path, ifstream::in);
	std::string model_name;
    fin >> model_name;
    cout << "model name is: " << model_name << endl;
	bool images_has_ground_truth = false;
	fin >> images_has_ground_truth;
	if (images_has_ground_truth) {
		cout << "the image lists must have ground_truth_shapes!\n" << endl;
	}
	else{
		cout << "the image lists does not have ground_truth_shapes!!!\n" << endl;
	}

	int path_num;
    fin >> path_num;
    cout << "reading testing images paths: " << endl;
	std::vector<std::string> image_path_prefixes;
    std::vector<std::string> image_lists;
    for (int i = 0; i < path_num; i++) {
        string s;
        fin >> s;
        cout << s << endl;
        image_path_prefixes.push_back(s);
        fin >> s;
        cout << s << endl;
        image_lists.push_back(s);
    }

	cout << "parsing config file done\n" << endl;
	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(model_name);
	cout << "load model done\n" << endl;
	std::vector<cv::Mat_<uchar> > images;
	std::vector<cv::Mat_<double> > ground_truth_shapes;
	std::vector<BoundingBox> bboxes;

	std::cout << "\nLoading test dataset..." << std::endl;
	if (images_has_ground_truth) {
		LoadImages(images, ground_truth_shapes, bboxes, image_path_prefixes, image_lists);
		double error = 0.0;
		for (int i = 0; i < images.size(); i++){
			cv::Mat_<double> current_shape = ReProjection(cas_load.params_.mean_shape_, bboxes[i]);
	        cv::Mat_<double> res = cas_load.Predict(images[i], current_shape, bboxes[i]);//, ground_truth_shapes[i]);
			double e = CalculateError(ground_truth_shapes[i], res);
			// std::cout << "error:" << e << std::endl;
			error += e;
	        // DrawPredictedImage(images[i], res);
		}
		std::cout << "error: " << error << ", mean error: " << error/images.size() << std::endl;
	}
	else{
		LoadImages(images, bboxes, image_path_prefixes, image_lists);
		for (int i = 0; i < images.size(); i++){
			cv::Mat_<double> current_shape = ReProjection(cas_load.params_.mean_shape_, bboxes[i]);
	        cv::Mat_<double> res = cas_load.Predict(images[i], current_shape, bboxes[i]);//, ground_truth_shapes[i]);
	        DrawPredictedImage(images[i], res);
		}
	}
}

void Train(const char* config_file_path){

	cout << "parsing config_file: " << config_file_path << endl;

    ifstream fin;
    fin.open(config_file_path, ifstream::in);
	std::string model_name;
    fin >> model_name;
    cout << "\nmodel name is: " << model_name << endl;
    Parameters params = Parameters();
    fin >> params.local_features_num_
        >> params.landmarks_num_per_face_
        >> params.regressor_stages_
        >> params.tree_depth_
        >> params.trees_num_per_forest_
        >> params.initial_guess_
		>> params.overlap_;

    std::vector<double> local_radius_by_stage;
    local_radius_by_stage.resize(params.regressor_stages_);
    for (int i = 0; i < params.regressor_stages_; i++){
            fin >> local_radius_by_stage[i];
    }
    params.local_radius_by_stage_ = local_radius_by_stage;
    params.output();

    int path_num;
    fin >> path_num;
    cout << "\nreading training images paths: " << endl;

	std::vector<std::string> image_path_prefixes;
    std::vector<std::string> image_lists;
    for (int i = 0; i < path_num; i++) {
        string s;
        fin >> s;
        cout << s << endl;
        image_path_prefixes.push_back(s);
        fin >> s;
        cout << s << endl;
        image_lists.push_back(s);
    }

    fin >> path_num;
    cout << "\nreading validation images paths: " << endl;
	std::vector<std::string> val_image_path_prefixes;
    std::vector<std::string> val_image_lists;
    for (int i = 0; i < path_num; i++) {
        string s;
        fin >> s;
        cout << s << endl;
        val_image_path_prefixes.push_back(s);
        fin >> s;
        cout << s << endl;
        val_image_lists.push_back(s);
    }

    cout << "parsing config file done\n" << endl;


	std::vector<cv::Mat_<uchar> > images;
	std::vector<cv::Mat_<double> > ground_truth_shapes;
	std::vector<BoundingBox> bboxes;

	std::vector<cv::Mat_<uchar> > val_images;
	std::vector<cv::Mat_<double> > val_ground_truth_shapes;
	std::vector<BoundingBox> val_bboxes;
	std::cout << "Loading training dataset..." << std::endl;
	LoadImages(images, ground_truth_shapes, bboxes, image_path_prefixes, image_lists);
	if (val_image_lists.size() > 0) {
		std::cout << "\nLoading validation dataset..." << std::endl;
		LoadImages(val_images, val_ground_truth_shapes, val_bboxes, val_image_path_prefixes, val_image_lists);
	}
	// else{
	// 	std::cout << "your validation dataset is 0" << std::endl;
	// }

	params.mean_shape_ = GetMeanShape(ground_truth_shapes, bboxes);
	CascadeRegressor cas_reg;
	cas_reg.val_bboxes_ = val_bboxes;
    cas_reg.val_images_ = val_images;
    cas_reg.val_ground_truth_shapes_ = val_ground_truth_shapes;

	cas_reg.Train(images, ground_truth_shapes, bboxes, params);
	std::cout << "finish training, start to saving the model..." << std::endl;
	std::cout << "model name: " << model_name << std::endl;
	cas_reg.SaveCascadeRegressor(model_name);
	std::cout << "save the model successfully\n" << std::endl;
}

void loadSelfDataFromText(std::vector<cv::Mat_<uchar> >& images,
              std::vector<cv::Mat_<double> >& ground_truth_shapes,
              std::vector<BoundingBox> & bounding_box, 
              std::string filepath, 
              int landmarkNum) {


    FILE* fin = fopen(filepath.c_str(), "r");
    if (fin == NULL){
       printf("%s is no exists", filepath.c_str());
       throw "load annotate data error";
       return;
    }
    std::cout << "global landmark num: " << landmarkNum << std::endl;
    std::string basename;
    if (landmarkNum == 74) {
        basename = "/home/sooda/data/photos/";
    } else {
        basename = "/home/sooda/data/lfpw/trainset/";
    }
    char filename[80];
    int cnt = 0;
    int i = 0; 
    while(fscanf(fin, "%s%*c", filename) != EOF) {
        i++;
        std::string fullname = basename + filename;
        std::cout << i << " " << fullname << std::endl;
        cv::Mat_<uchar>  temp = cv::imread(fullname, 0);
        images.push_back(temp);

        int left, top, right, bottom;
        fscanf(fin, "%d %d %d %d%*c", &left, &top, &right, &bottom);

        BoundingBox bb;
        bb.start_x = left;
        bb.start_y = top;
        bb.width = right - left;
        bb.height = bottom - top;
        bb.center_x = bb.start_x + bb.width / 2.0;
        bb.center_y = bb.start_y + bb.height / 2.0;
        bounding_box.push_back(bb);

        cv::Mat_<double> ptsmat(landmarkNum, 2);
        for(int j = 0; j < landmarkNum; j++) { 
            int x, y;
            fscanf(fin, "%d %d%*c", &x, &y);
            ptsmat(j, 0) = x;
            ptsmat(j, 1) = y;
            //cv::circle(temp, cv::Point2d(x, y),3, cv::Scalar(255, 0, 255), -1);
        }
        //cv::imshow("temp", temp);
        //cv::waitKey();
        ground_truth_shapes.push_back(ptsmat);
    }
    fclose(fin);
    assert(ground_truth_shapes.size() == bounding_box.size());
}

void TrainSelfText(const char* annoName, const char* ModelName, int landmarkNum = 68){
    std::cout << "annoName: " << annoName << " ModelName: " << ModelName << " " << landmarkNum << std::endl;
	std::vector<cv::Mat_<uchar> > images;
	std::vector<cv::Mat_<double> > ground_truth_shapes;
	std::vector<BoundingBox> bboxes;
    //std::string file_names = "./../dataset/helen/train_jpgs.txt";
    std::string file_names(annoName);
	//LoadImages(images, ground_truth_shapes, bboxes, file_names);
    loadSelfDataFromText(images, ground_truth_shapes, bboxes, file_names, landmarkNum);

	Parameters params;
    params.local_features_num_ = 300;
	params.landmarks_num_per_face_ = landmarkNum;
    params.regressor_stages_ = 6;
	params.local_radius_by_stage_.push_back(0.4);
    params.local_radius_by_stage_.push_back(0.3);
    params.local_radius_by_stage_.push_back(0.2);
	params.local_radius_by_stage_.push_back(0.1);
    params.local_radius_by_stage_.push_back(0.08);
    params.local_radius_by_stage_.push_back(0.05);
    params.tree_depth_ = 5;
    params.trees_num_per_forest_ = 8;
    params.initial_guess_ = 5;

	params.mean_shape_ = GetMeanShape(ground_truth_shapes, bboxes);
	CascadeRegressor cas_reg;
	cas_reg.Train(images, ground_truth_shapes, bboxes, params);
	cas_reg.SaveCascadeRegressor(ModelName);
	return;
}



std::vector<cv::Rect> dlibFaceDetect(frontal_face_detector detector, cv::Mat gray) { 
    dlib::array2d<unsigned char> img;
    dlib::cv_image<unsigned char> *pimg = new dlib::cv_image<unsigned char>(gray);
    assign_image(img, *pimg);
    delete pimg;
    std::vector<dlib::rectangle> dets;
    dets  = detector(img);
    std::vector<cv::Rect> faces;
    for(int i = 0; i < dets.size(); i++) {
        cv::Rect rect = cv::Rect(cv::Point(dets[i].left(), dets[i].top()), cv::Point(dets[i].right(), dets[i].bottom()) );
        faces.push_back(rect);
    }
    return faces;
}

void dlibTestImage(const char* name, CascadeRegressor& rg, frontal_face_detector detector) {
    cv::Mat src = cv::imread(name, 0);
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, CV_BGR2GRAY);
    } else if(src.channels() == 1) {
        gray = src;
    }
    struct timeval t1, t2;
    double timeuse;
    std::vector<cv::Rect> faces = dlibFaceDetect(detector, gray);
    gettimeofday(&t2, NULL);
    cv::Mat_<uchar> image = gray;
    cout << "dlib face detected " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
    for (int i = 0; i < faces.size(); i++){
        cv::Rect faceRec = faces[i];
        BoundingBox bbox;
        bbox.start_x = faceRec.x;
        bbox.start_y = faceRec.y;
        bbox.width = faceRec.width;
        bbox.height = faceRec.height;
        bbox.center_x = bbox.start_x + bbox.width / 2.0;
        bbox.center_y = bbox.start_y + bbox.height / 2.0;
        cv::Mat_<double> current_shape = ReProjection(rg.params_.mean_shape_, bbox);
        gettimeofday(&t1, NULL);
        cv::Mat_<double> res = rg.Predict(image, current_shape, bbox);//, ground_truth_shapes[i]);
        gettimeofday(&t2, NULL);
        cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
        cv::rectangle(image, faceRec, (255), 1);
        for (int i = 0; i < res.rows; i++){
            cv::circle(image, cv::Point2f(res(i, 0), res(i, 1)), 2, (255));
        }
    }
    cv::imshow("detect", image);
    cv::waitKey();
}

void batchTest(const char* ModelName, int argc, char* argv[]) {
	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(ModelName);
    std::cout << "load ok" << std::endl;
    frontal_face_detector detector;
    string dlib_face_detector = "../data/front_face.dat";
    dlib::deserialize(dlib_face_detector) >> detector;
    for(int i = 3; i < argc; i++) {
        //TestImage(argv[i], cas_load);
        std::cout << argv[i] << std::endl;
        dlibTestImage(argv[i], cas_load, detector);
    }
}

int main(int argc, char* argv[])
{
	std::cout << "\nuse [./application train train_config_file] to train models" << std::endl;
	std::cout << "    [./application test test_config_file] to test images\n\n" << std::endl;


	if (strcmp(argv[1], "train") == 0)
	{
		Train(argv[2]);
        return 0;
	} 
	else if (strcmp(argv[1], "test") == 0)
	{
		Test(argv[2]);
        return 0;
	}
	else if (strcmp(argv[1], "traintxt") == 0) 
	{ 
		//usage: ./application traintxt ~/data/photos/total_xx.txt modelname 74
        int landmark_num = atoi(argv[4]);
        std::cout << "train txt " << landmark_num << std::endl;
        TrainSelfText(argv[2],argv[3], landmark_num);
		return 0;
	}
	else if (strcmp(argv[1], "demo") == 0) 
	{
		//usage: ./application demo modelName ~/data/face1/*.jpg
		batchTest(argv[2], argc, argv);
	}

	return 0;
}
