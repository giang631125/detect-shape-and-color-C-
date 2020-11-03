#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main( ) {
    int     x1, y1, x2, y2, w, h, temp=0, key, contour_size;
    String  file_name;
    int     shape_type;
    Mat     image, im_tranf;         
    image = imread("shape-train.png" ,1); 
    cvtColor(image, im_tranf, CV_BGR2GRAY);
    GaussianBlur(im_tranf,im_tranf,Size(3,3),11);
    threshold(im_tranf, im_tranf, 220, 255, THRESH_BINARY_INV);    //    imshow("thresh", im_tranf);
    Canny(im_tranf, im_tranf, 20, 200,5);                          //    imshow("canny", im_tranf);
    vector<vector<Point>>   contour;
    vector<Vec4i>           hierarchy;
    findContours(im_tranf, contour, hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    contour_size = contour.size();
    Mat                     im_roi, im_roi_small;
    vector<vector<Point>>   contour_poly(contour_size);
    vector<Rect>            boundRect( contour_size );
    for(int i=0; i<contour_size; i++)
    {
        cout<<"area "<<i<<" "<<contourArea( contour[i] )<<endl;
        if( contourArea(contour[i]) > 10)
        {
            approxPolyDP( contour[i], contour_poly[i], 3, true );
            boundRect[i] = boundingRect( contour_poly[i] );
            x1 = boundRect[i].tl().x;
            y1 = boundRect[i].tl().y;
            x2 = boundRect[i].br().x;
            y2 = boundRect[i].br().y;
            w = x2-x1;
            h = y2-y1;
            if( h>5 )
            {
                temp++;
                rectangle( image, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0),1 );
                Rect rect1(x1+1, y1+1, w-1, h-1);
                im_roi = image(rect1);
                resize( im_roi, im_roi_small, Size(30,30), 0, 0);
                imshow("image roi", im_roi);
                cout<<"Import shape: ";
                cin>>shape_type;
                switch (shape_type)
                {
                case 1:
                    file_name = format( "file-image/circle/%d.png", temp );
                    break;
                case 2:
                    file_name = format( "file-image/rectangle/%d.png", temp );
                    break;
                case 3:
                    file_name = format( "file-image/triangle/%d.png", temp );
                    break;
                default:
                    break;
                }
                imshow("image doing", image);                
                imwrite( file_name, im_roi);
            }
        }
    }

    waitKey(0);
    destroyAllWindows();
    return 0;
}