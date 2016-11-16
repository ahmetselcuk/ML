         
#pragma comment(lib, "opencv_core300d.lib")       
#pragma comment(lib, "opencv_highgui300d.lib")    
#pragma comment(lib, "opencv_imgcodecs300d.lib")  
#pragma comment(lib, "opencv_videoio300d.lib")  
#pragma comment(lib, "opencv_imgproc300d.lib")  
#pragma comment(lib, "opencv_ml300d.lib") 



#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;

void loadFromCSV(const std::string& filename , std::vector< std::vector<std::string> >&   matrix , char del = ',');
void readData(const char *fName , cv::Mat& Data , cv::Mat& Labels);
bool trainSVM(const char *fName , cv::Mat& Data , cv::Mat& Labels);
void testSVM(const char *fName , cv::Mat& Data , cv::Mat& Labels);

bool trainBayes(cv::Ptr <cv::ml::NormalBayesClassifier > bayes , cv::Mat& Data , cv::Mat& Labels);
void testBayes(cv::Ptr <cv::ml::NormalBayesClassifier > bayes , cv::Mat& Data , cv::Mat& Labels);

bool trainBayes(const char *fName , cv::Mat& Data , cv::Mat& Labels);
void testBayes(const char *fName , cv::Mat& Data , cv::Mat& Labels);

bool trainKNearest(cv::Ptr<cv::ml::KNearest> kNearest , cv::Mat& Data , cv::Mat& Labels);
void testKNearest(cv::Ptr<cv::ml::KNearest> kNearest , cv::Mat& Data , cv::Mat& Labels);


bool trainKNearest(const char *fName , cv::Mat& Data , cv::Mat& Labels);
void testKNearest(const char *fName , cv::Mat& Data , cv::Mat& Labels);

bool trainDTree(const char *fName , cv::Mat& Data , cv::Mat& Labels);
void testDTree(const char *fName , cv::Mat& Data , cv::Mat& Labels);

bool trainRTree(const char *fName , cv::Mat& Data , cv::Mat& Labels);
void testRTree(const char *fName , cv::Mat& Data , cv::Mat& Labels);

bool trainANN(const char *fName , cv::Mat& Data , cv::Mat& Labels);
void testANN(const char *fName , cv::Mat& Data , cv::Mat& Labels);

void UnsupervisedEMDemo();


int main()
	{

	const char *fNameTrain = "E:/CppCodes/ML/Kuyumcu/datas/ml/irisTrain.txt";
	const char *fNameTest = "E:/CppCodes/ML/Kuyumcu/datas/ml/irisTest.txt";
	const char *fNameSvmModel = "E:/CppCodes/ML/Kuyumcu/datas/ml/irisSvm.xml";
	const char *fNameBayesModel = "E:/CppCodes/ML/Kuyumcu/datas/ml/irisBayes.xml";
	const char *fNameKNearestModel = "E:/CppCodes/ML/Kuyumcu/datas/ml/irisKNearest.xml";
	const char *fNameDTreeModel = "E:/CppCodes/ML/Kuyumcu/datas/ml/irisDTree.xml";
	const char *fNameRTreeModel = "E:/CppCodes/ML/Kuyumcu/datas/ml/irisRTree.xml";
	const char *fNameANNModel = "E:/CppCodes/ML/Kuyumcu/datas/ml/irisANN.xml";
	const char *fNameLRegModel = "E:/CppCodes/ML/Kuyumcu/datas/ml/irisLReg.xml";



	cv::Mat trainData , trainLabels;
	cv::Mat testData , testLabels;

	readData(fNameTrain , trainData , trainLabels);
	readData(fNameTest , testData , testLabels);


	if ( trainBayes(fNameBayesModel , trainData , trainLabels) )
		testBayes(fNameBayesModel , testData , testLabels);

	if ( trainSVM(fNameSvmModel , trainData , trainLabels) )
		testSVM(fNameSvmModel , testData , testLabels);

	if ( trainDTree(fNameDTreeModel , trainData , trainLabels) )
		testDTree(fNameDTreeModel , testData , testLabels);

	if ( trainRTree(fNameRTreeModel , trainData , trainLabels) )
		testRTree(fNameRTreeModel , testData , testLabels);

	if ( trainKNearest(fNameKNearestModel , trainData , trainLabels) )
		testKNearest(fNameKNearestModel , testData , testLabels);

	if ( trainANN(fNameANNModel , trainData , trainLabels) )
		testANN(fNameANNModel , testData , testLabels);

	UnsupervisedEMDemo();

	return 0;
	}

void loadFromCSV(const std::string& filename , std::vector< std::vector<std::string> >&   matrix , char del)
	{
	std::ifstream       file(filename.c_str());
	std::vector<std::string>   row;
	std::string                line;
	std::string                cell;

	cout << "loadFromCSV : " << filename << endl;
	while ( file )
		{
		std::getline(file , line);
		std::stringstream lineStream(line);
		row.clear();

		while ( std::getline(lineStream , cell , del) )
			row.push_back(cell);

		if ( !row.empty() )
			matrix.push_back(row);
		}

	}

void readData(const char *fName , cv::Mat& Data , cv::Mat& Labels)
	{
	std::vector< std::vector<std::string> >   matrix;
	const char *irisNames[3] = { "Iris-setosa" ,
		"Iris-versicolor" ,
		"Iris-virginica"
		};

	loadFromCSV(fName , matrix);
	Data = cv::Mat(matrix.size() , 4 , CV_32FC1);
	Labels = cv::Mat(matrix.size() , 1 , CV_32SC1);

	for ( int i = 0; i<int(matrix.size()); i++ )
		{
		for ( int j = 0; j<4; j++ )
			{

			Data.at<float>(i , j) = atof(matrix[i][j].c_str());
			}
		float labelId;
		if ( matrix[i][4] == irisNames[0] )
			labelId = 0.0;
		else if ( matrix[i][4] == irisNames[1] )
			labelId = 1.0;
		else // if (matrix[i][4]== irisNames[2])
			labelId = 2.0;

		Labels.at<int>(i , 0) = labelId;
		}

	}


bool trainBayes(const char *fName , cv::Mat& Data , cv::Mat& Labels)
	{

	cv::Ptr <cv::ml::NormalBayesClassifier > bayes = cv::ml::NormalBayesClassifier::create();

	bool isTrained = false;

	isTrained = bayes->train(Data , cv::ml::ROW_SAMPLE , Labels);

	if ( isTrained )
		{
		cout << "succesfull" << endl;
		bayes->save(fName);
		}
	else
		cout << "Not succesfull" << endl;

	return isTrained;

	}

void testBayes(const char *fName , cv::Mat& Data , cv::Mat& Labels)
	{
	cout << "\n\nTesting Bayes : " << fName << endl;

	cv::Ptr <cv::ml::NormalBayesClassifier > bayes = cv::Algorithm::load<cv::ml::NormalBayesClassifier>(fName);

	if ( bayes->empty() )
		{
		cout << " empty algorithm Model cannot loaded" << endl;
		return;
		}

	int confMatrix[3][3] = { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 };

	for ( int i = 0; i<Labels.rows; i++ )
		{
		float out;
		out = bayes->predict(Data.row(i));
		cout << i + 1 << ") " << Labels.at<int>(i) << " is Predicted  as : " << out << endl;
		confMatrix[( int )Labels.at<int>(i)][int(out)]++;
		}
	cout << "  Confusion Matrix   " << endl;
	for ( int i = 0; i<3; i++ )
		cout << "    " << confMatrix[i][0] << " " << confMatrix[i][1] << " " << confMatrix[i][2] << endl;
	}

bool trainSVM(const char *fName , cv::Mat& Data , cv::Mat& Labels)
	{
	bool isParamAuto = false;
	bool isTrained = false;

	cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(Data , cv::ml::ROW_SAMPLE , Labels);

	cv::Ptr<cv::ml::SVM > svm = cv::ml::SVM::create();

	if ( isParamAuto == true )
		{
		isTrained = svm->trainAuto(tData);
		}
	else
		{
		svm->setType(cv::ml::SVM::Types::C_SVC);
		svm->setKernel(cv::ml::SVM::KernelTypes::POLY);
		svm->setDegree(1);
		svm->setGamma(20);
		svm->setCoef0(1);

		isTrained = svm->train(Data , cv::ml::ROW_SAMPLE , Labels);
		}

	if ( isTrained )
		{
		cout << "succesfull" << endl;
		svm->save(string(fName));
		}
	else
		cout << "Not succesfull" << endl;

	return isTrained;

	}

void testSVM(const char *fName , cv::Mat& Data , cv::Mat& Labels)
	{
	cout << "\n\nTesting Svm : " << fName << endl;

	cv::Ptr<cv::ml::SVM > svm = cv::Algorithm::load<cv::ml::SVM>(fName);

	if ( svm->empty() )
		{
		cout << " empty algorithm Model cannot loaded" << endl;
		return;
		}

	int confMatrix[3][3] = { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 };
	for ( int i = 0; i<Labels.rows; i++ )
		{
		float out;
		out = svm->predict(Data.row(i));
		cout << i + 1 << ") " << Labels.at<int>(i) << " is Predicted  as : " << out << endl;
		confMatrix[( int )Labels.at<int>(i)][int(out)]++;
		}
	cout << "  Confusion Matrix   " << endl;
	for ( int i = 0; i<3; i++ )
		cout << "    " << confMatrix[i][0] << " " << confMatrix[i][1] << " " << confMatrix[i][2] << endl;
	}

bool trainDTree(const char *fName , cv::Mat& Data , cv::Mat& Labels)
	{

	cout << "\n\nTraining DTree : " << endl;

	cv::Ptr <cv::ml::DTrees> dTree = cv::ml::DTrees::create();

	bool isTrained = false;


	dTree->setMaxDepth(8);
	dTree->setMinSampleCount(2);
	dTree->setCVFolds(1);


	isTrained = dTree->train(Data , cv::ml::ROW_SAMPLE , Labels);


	if ( isTrained )
		{
		cout << "succesfull" << endl;
		dTree->save(fName);
		}
	else
		cout << "Not succesfull" << endl;

	return isTrained;

	}

void testDTree(const char *fName , cv::Mat& Data , cv::Mat& Labels)
	{
	cout << "\n\nTesting DTree : " << fName << endl;

	cv::Ptr <cv::ml::DTrees> dTree = cv::Algorithm::load<cv::ml::DTrees>(fName);

	if ( dTree->empty() )
		{
		cout << " empty algorithm Model cannot loaded" << endl;
		return;
		}


	int confMatrix[3][3] = { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 };

	for ( int i = 0; i<Labels.rows; i++ )
		{
		float out;
		out = dTree->predict(Data.row(i));
		cout << i + 1 << ") " << Labels.at<int>(i) << " is Predicted  as : " << out << endl;
		confMatrix[( int )Labels.at<int>(i)][int(out)]++;
		}
	cout << "  Confusion Matrix   " << endl;
	for ( int i = 0; i<3; i++ )
		cout << "    " << confMatrix[i][0] << " " << confMatrix[i][1] << " " << confMatrix[i][2] << endl;

	}

bool trainRTree(const char *fName , cv::Mat& Data , cv::Mat& Labels)
	{

	cout << "\n\nTraining RTrees : " << endl;

	bool isTrained = false;

	cv::Ptr <cv::ml::RTrees> rTree = cv::ml::RTrees::create();
	rTree->setCalculateVarImportance(true);

	isTrained = rTree->train(Data , cv::ml::ROW_SAMPLE , Labels);


	if ( isTrained )
		{
		cout << "succesfull" << endl;
		rTree->save(fName);
		}
	else
		cout << "Not succesfull" << endl;

	return isTrained;

	}

void testRTree(const char *fName , cv::Mat& Data , cv::Mat& Labels)
	{
	cout << "\n\nTesting RTrees : " << fName << endl;

	cv::Ptr <cv::ml::RTrees> rTree = cv::Algorithm::load<cv::ml::RTrees>(fName);

	if ( rTree->empty() )
		{
		cout << " empty algorithm Model cannot loaded" << endl;
		return;
		}


	int confMatrix[3][3] = { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 };

	for ( int i = 0; i<Labels.rows; i++ )
		{
		float out;
		out = rTree->predict(Data.row(i) , cv::noArray());
		cout << i + 1 << ") " << Labels.at<int>(i) << " is Predicted  as : " << out << endl;
		confMatrix[( int )Labels.at<int>(i)][int(out)]++;
		}
	cout << "  Confusion Matrix   " << endl;
	for ( int i = 0; i<3; i++ )
		cout << "    " << confMatrix[i][0] << " " << confMatrix[i][1] << " " << confMatrix[i][2] << endl;


	cv::Mat vIm = rTree->getVarImportance();

	cout << " Variable importance array  :\n" << vIm << endl;

	}

bool trainKNearest(const char *fName , cv::Mat& Data , cv::Mat& Labels)
	{


	bool isTrained = false;

	cv::Ptr<cv::ml::KNearest> kNearest = cv::ml::KNearest::create();

	cout << "KNN training started " << endl;
	isTrained = kNearest->train(Data , cv::ml::ROW_SAMPLE , Labels);

	if ( isTrained )
		{
		cout << "succesfull" << endl;
		kNearest->save(fName);
		}
	else
		cout << "Not succesfull" << endl;

	return isTrained;

	}

void testKNearest(const char *fName , cv::Mat& Data , cv::Mat& Labels)
	{
	cout << "\n\nTesting KNearest : " << endl;

	cv::Ptr<cv::ml::KNearest> kNearest = cv::Algorithm::load<cv::ml::KNearest>(fName);


	if ( kNearest->empty() )
		{
		cout << " empty algorithm Model cannot loaded" << endl;
		return;
		}


	int confMatrix[3][3] = { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 };
	cv::Mat nResponce , distances;

	for ( int i = 0; i<Labels.rows; i++ )
		{
		float out;

		// out=kNearest->findNearest(Data.row(i),3,cv::noArray()); //

		out = kNearest->findNearest(Data.row(i) , 3 , cv::noArray() , nResponce , distances);

		cout << i + 1 << ") " << Labels.at<int>(i) << " is Predicted  as : " << out << endl;

		cout << "neighbor Responses \n" << nResponce << endl;
		cout << "Distance \n" << distances << endl;

		confMatrix[( int )Labels.at<int>(i)][int(out)]++;
		}
	cout << "  Confusion Matrix   " << endl;
	for ( int i = 0; i<3; i++ )
		cout << "    " << confMatrix[i][0] << " " << confMatrix[i][1] << " " << confMatrix[i][2] << endl;
	}

bool trainANN(const char *fName , cv::Mat& Data , cv::Mat& Labels)
	{

	cout << "ANN training started " << endl;

	// Labels to Outputs
	cv::Mat outputs = cv::Mat::zeros(Labels.rows , 3 , CV_32FC1);


	for ( int i = 0; i<Labels.rows; i++ )
		{
		int id = Labels.at<int>(i);
		outputs.at<float>(i , id) = 1;
		}


	cv::Mat layerSizes(1 , 3 , CV_32SC1);

	layerSizes.at<int>(0) = Data.cols; // Input Layer
	layerSizes.at<int>(1) = 5; // Hidden Layer
	layerSizes.at<int>(2) = 3; // Output layer

	cv::Ptr <cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
	ann->setLayerSizes(layerSizes);
	ann->setActivationFunction(cv::ml::ANN_MLP::GAUSSIAN , 1 , 1);
	ann->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS , 300 , FLT_EPSILON));
	ann->setTrainMethod(cv::ml::ANN_MLP::BACKPROP , 0.001);

	bool isTrained = false;
	isTrained = ann->train(Data , cv::ml::ROW_SAMPLE , outputs);

	if ( isTrained )
		{
		cout << "succesfull" << endl;
		ann->save(fName);
		}
	else
		cout << "Not succesfull" << endl;

	return isTrained;

	}

void testANN(const char *fName , cv::Mat& Data , cv::Mat& Labels)
	{

	cout << "\n\nTesting ANN : " << endl;

	int confMatrix[3][3] = { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 };

	cv::Ptr <cv::ml::ANN_MLP> ann = cv::Algorithm::load<cv::ml::ANN_MLP>(fName);

	if ( ann->empty() )
		{
		cout << " empty algorithm Model cannot loaded" << endl;
		return;
		}


	for ( int i = 0; i<Labels.rows; i++ )
		{
		cv::Mat out(1 , 3 , CV_32FC1);
		float f;
		f = ann->predict(Data.row(i) , out);
		cout << "Out size " << out.size() << " return value " << f << endl;
		cv::Point maxLoc;
		double maxVal;

		cv::minMaxLoc(out , 0 , &maxVal , 0 , &maxLoc);
		cout << i + 1 << ") " << Labels.at<int>(i) << " is Predicted  as : " << maxLoc.x << " Full Output " << out << endl;
		confMatrix[( int )Labels.at<int>(i)][maxLoc.x]++;
		}
	cout << "  Confusion Matrix   " << endl;
	for ( int i = 0; i<3; i++ )
		cout << "    " << confMatrix[i][0] << " " << confMatrix[i][1] << " " << confMatrix[i][2] << endl;


	}


void UnsupervisedEMDemo()
	{
	const char *fNameEMModel = "E:/CppCodes/ML/Kuyumcu/datas/ml/Em.xml";

	int nRegion = 3;
	bool withLocation = false;

	int nInput = 3;
	if ( withLocation == true )
		nInput = 5;

	cv::Mat labImg;

	cout << "Unsupervised Expectation Maximization Demo" << endl;

	cv::Mat inImg = cv::imread("E:/CppCodes/ML/Kuyumcu/datas/BloodCells.jpg"); // //uydu.jpg
	cv::imshow("Orj" , inImg);
	cv::waitKey(1);

	cv::cvtColor(inImg , labImg , CV_BGR2Lab);
	cv::pyrDown(labImg , labImg);



	cv::Mat Data = cv::Mat(labImg.cols*labImg.rows , nInput , CV_32FC1);

	int i = 0;
	for ( int x = 0; x<labImg.cols; x++ )
		{
		for ( int y = 0; y<labImg.rows; y++ )
			{
			for ( int k = 0; k<3; k++ )
				{
				Data.at<float>(i , k) = labImg.at<cv::Vec3b>(y , x)[k];
				}
			if ( withLocation == true )
				{
				Data.at<float>(i , 3) = x;
				Data.at<float>(i , 4) = y;
				}
			i++;
			}
		}

	cv::Ptr <cv::ml::EM> em = cv::ml::EM::create();
	em->setClustersNumber(nRegion);

	cout << "Started to Training EM " << endl;

	bool isTrained;

	isTrained = em->trainEM(Data);
	if ( isTrained )
		{
		cout << "\nEM succesfully Trained" << endl;
		em->save(fNameEMModel);
		}
	else
		cout << "\nEM Training Failed..." << endl;


	// Testing 

	em->clear();
	em = cv::Algorithm::load<cv::ml::EM>(fNameEMModel);

	cv::Mat outImg = cv::Mat::zeros(labImg.size() , labImg.type());

	for ( int i = 0; i<Data.rows; i++ )
		{
		int regionId = em->predict(Data.row(i));
		int x = i / labImg.rows;
		int y = ( i%labImg.rows );

		if ( regionId == 0 )
			{
			outImg.at<cv::Vec3b>(y , x) = cv::Vec3b(255 , 0 , 0);

			}
		else if ( regionId == 1 )
			{
			outImg.at<cv::Vec3b>(y , x) = cv::Vec3b(0 , 255 , 0);
			}
		else if ( regionId == 2 )
			{
			outImg.at<cv::Vec3b>(y , x) = cv::Vec3b(0 , 0 , 255);
			}
		else if ( regionId == 3 )
			{
			outImg.at<cv::Vec3b>(y , x) = cv::Vec3b(255 , 255 , 0);
			}
		else if ( regionId == 4 )
			{
			outImg.at<cv::Vec3b>(y , x) = cv::Vec3b(255 , 0 , 255);
			}
		else if ( regionId == 5 )
			{
			outImg.at<cv::Vec3b>(y , x) = cv::Vec3b(0 , 255 , 255);
			}

		}
	cv::imwrite("E:/CppCodes/ML/Kuyumcu/datas/out12.jpg" , outImg);
	cv::pyrUp(outImg , outImg);


	cv::imshow("Orj" , inImg);
	cv::imshow("EM out" , outImg);
	cv::waitKey(0);
	cv::destroyAllWindows();
	}


