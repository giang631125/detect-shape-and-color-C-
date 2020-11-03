#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>      // std::setw()
using namespace std;
using namespace cv::ml;
using namespace cv;
// g++ training.cpp -o training `pkg-config --cflags --libs opencv` && ./training
#define size_image 10
#define num_color 7
#define file_name_image_test    "shape-test-cap.jpg"

void convert_to_1d(Mat_<float> Mat_2d, Mat_<float> Mat_out)
{
    for(int i=0; i<size_image; i++)
    {
        Mat_2d.row(i).copyTo(Mat_out(Rect(size_image*i, 0, size_image, 1)));
    }
}
void get_image(int shape, String text, int *stt_image, Mat1f* image_table, vector<int>* label)
{
    vector<String> file_name;
    glob( text, file_name, false );
    int num_image = file_name.size();
    Mat1f image_readed;
    for (int j=0; j<num_image; j++)
    {
        (*stt_image)++;
        image_readed = imread( file_name[j], 0);
        cout<<*stt_image<<" "<<image_readed.size()<<endl;
        resize(image_readed, image_readed, Size(size_image, size_image),0,0);
        (*image_table).push_back(image_readed);
        (*label).push_back(shape);
    }
}
int main( ) {
    ofstream mySampleFile, myTestFile, myTrainingFile; 
    mySampleFile.open("file-text/sample.txt", std::ofstream::out | std::ofstream::trunc);           // clear all data of file
    myTrainingFile.open("file-text/training.txt", std::ofstream::out | std::ofstream::trunc);       // clear all data of file
    myTestFile.open("file-text/test.txt", std::ofstream::out | std::ofstream::trunc);               // clear all data of file

    // ------------------------------------read all image-------------------------------------
    vector<String> file_name;
    Mat_<float> image_table=Mat(0,0,CV_8UC1);
    vector<int> label_shape_vector;
    int i, stt_image = 0;
    String text;

    get_image(1, "file-image/circle/*.png", &stt_image, &image_table, &label_shape_vector);
    get_image(2, "file-image/rectangle/*.png", &stt_image, &image_table, &label_shape_vector);
    get_image(3, "file-image/triangle/*.png", &stt_image, &image_table, &label_shape_vector);
    cout<<"sum image: "<<stt_image<<endl;
    // cout<<"size of image table: "<<image_table.size()<<endl;

    //--------------------------------------training shape data------------------------------------
    Mat_<float> train_shape_data = Mat::zeros(stt_image, size_image*size_image, CV_8UC1);
    for(i=0; i< size_image*stt_image; i++)                             // convert matrix 2d to matrix 1d
    {
        image_table.row(i).copyTo(train_shape_data(cvRect(size_image*(i%size_image),i/size_image,size_image,1)));
    }    
    myTrainingFile  <<train_shape_data    <<endl;

    Mat_<int> label_shape_Mat = Mat::zeros( 1, stt_image, CV_8UC1 );
    memcpy( label_shape_Mat.data, label_shape_vector.data(), label_shape_vector.size()*sizeof(int) );
    mySampleFile    <<"label shape"<<endl<<label_shape_Mat    <<endl;

    Ptr<KNearest> knear_shape(KNearest::create());
    knear_shape->train(train_shape_data, ROW_SAMPLE, label_shape_Mat);
    cout<<"Training shape complete."<<endl;

    //--------------------------------------training color data------------------------------------
    Mat_<float> train_color_data = Mat::zeros(num_color, 3, CV_8UC1);
    Mat1f image_color_table = Mat(0,0,CV_8UC1);
    Mat image_color = imread("color_train.png");
    Vec3b pixel;
    vector<Vec3b> pixel_vector;
    for(i=0; i<num_color; i++)
    {
        pixel = image_color.at<Vec3b>(5, 10*i+5);
        pixel_vector.push_back(pixel);
    }
    for(int j=0; j<num_color; j++)
    {
        for(i=0; i<3; i++)
        {
            image_color_table.push_back(pixel_vector.at(j).val[2-i]);
        }
    }
    for(int j=0; j<num_color; j++)
    {
        for(i=0; i<3; i++)
        {
            image_color_table.row(3*j+i).copyTo(train_color_data(cvRect(i,j,1,1)));
        }
    }
    myTrainingFile<<"training color"<<endl;
    myTrainingFile<<train_color_data<<endl;

    Mat_<int> label_color_Mat = Mat::zeros( 1, num_color, CV_8UC1 );
    vector<int> label_color_vector;
    for(i=0; i<num_color; i++)
    {
        label_color_vector.push_back(i+1);
    }
    memcpy( label_color_Mat.data, label_color_vector.data(), label_color_vector.size()*sizeof(int) );
    mySampleFile <<"label color"<<endl<<label_color_Mat<<endl;

    Ptr<KNearest> knear_color(KNearest::create());
    knear_color->train(train_color_data, ROW_SAMPLE, label_color_Mat);
    cout<<"Training color complete."<<endl;
    
    //-----------------------------------------test data-------------------------------------
    Mat image_test_origin, image_test_threshold, image_test_canny;
    image_test_origin = imread(file_name_image_test, CV_LOAD_IMAGE_COLOR);
    resize(image_test_origin, image_test_origin, Size(500, 400),0,0);
    cvtColor(image_test_origin, image_test_threshold, CV_BGR2GRAY);
    GaussianBlur(image_test_threshold, image_test_threshold, Size(3,3), 11);
    threshold(image_test_threshold, image_test_threshold, 160, 255, THRESH_BINARY);  imshow("thresh", image_test_threshold);
    Canny(image_test_threshold, image_test_canny, 50, 200, 5);                       imshow("canny", image_test_canny);

    vector<vector<Point>> test_contour;
    vector<Vec4i> test_hierarchy;
    findContours(image_test_canny, test_contour, test_hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    int test_contour_size = test_contour.size();
    cout<<"contour test size: "<< test_contour_size<<endl;

    Mat image_roi_test, image_roismall_test, image_roi_color_test;
    Mat resu_shape, dist_shape, resu_color, dist_color;
    Mat1f image_test_float_1d = Mat::zeros(1, size_image*size_image,CV_8UC1);
    Mat_<float> test_color_data = Mat::zeros(1, 3, CV_8UC1);

    Mat1f image_color_test = Mat(0,0,CV_8UC1);
    int x1, y1, x2, y2, w, h, K=1, key;
    for(int i2=0; i2<test_contour_size; i2++)
    {
        if( contourArea(test_contour[i2])>500 )
        {
            vector<vector<Point> > contours_poly( test_contour_size );
            approxPolyDP( test_contour[i2], contours_poly[i2], 3, true );
            vector<Rect> boundRect(test_contour_size );                          // bound = rang buoc
            boundRect[i2] = boundingRect( contours_poly[i2] );
            x1 = boundRect[i2].tl().x;  y1 = boundRect[i2].tl().y;
            x2 = boundRect[i2].br().x;  y2 = boundRect[i2].br().y;
            w = x2 - x1;                h = y2 - y1;
            if (h>20)
            {
                //draw rectangle around contours
                rectangle( image_test_origin, Point(x1,y1), Point(x2,y2), Scalar(0,255,0), 1);
                Rect rect_test(x1+1,y1+1,w-1,h-1);
                image_roi_test = image_test_threshold(rect_test);                          // crop origin image
                image_roi_color_test = image_test_origin(rect_test);
                Mat1f image_color_test;                                                     // renew
                pixel = image_roi_color_test.at<Vec3b>(h/2, w/2);
                for(i=0; i<3; i++)
                {
                    image_color_test.push_back(pixel.val[2-i]);
                }
                for(i=0; i<3; i++)
                {
                    image_color_test.row(i).copyTo(test_color_data(cvRect(i,0,1,1)));
                }
                myTestFile<<test_color_data<<endl;
                resize(image_roi_test, image_roismall_test, Size(size_image,size_image), 0, 0);
            
                Mat1f image_test_float;
                image_test_float.push_back(image_roismall_test);
                convert_to_1d(image_test_float, image_test_float_1d);
                knear_shape->findNearest(image_test_float_1d, K, noArray(), resu_shape, dist_shape );     // find label of image
                cout <<" result shape:"<<resu_shape.row(0)<<"   distance:"<<dist_shape;
                knear_color->findNearest(test_color_data, K, noArray(), resu_color, dist_color);
                cout <<setw(30)<<" result color:"<<resu_color.row(0)<<"   distance:"<<dist_color<< endl;
                
                imshow("image test roi", image_roi_test);
                imshow("image testing", image_test_origin);
                key = waitKey(0);
                if (key == 27)  // ESC key to quit
                    break;
            }
        }
    }
    myTestFile<<image_test_float_1d<<endl;
    cout<<"test done"<<endl;
    waitKey(0);
    mySampleFile.close();    
    myTrainingFile.close();
    myTestFile.close();
    destroyAllWindows();
    return 0;
}