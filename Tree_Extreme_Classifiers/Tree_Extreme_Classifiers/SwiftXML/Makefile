INC=-I../Tools/c++ -I../FastXML -I../PfastreXML
CXXFLAGS=-std=c++11 -O3 -g
LIBFLAGS=-pthread

all: clean swiftXML_train swiftXML_predict

swiftXML_train:
	g++ -o swiftXML_train $(CXXFLAGS) $(INC) swiftXML_train.cpp swiftXML.cpp ../FastXML/fastXML.cpp ../Tools/c++/mat.cpp $(LIBFLAGS)

swiftXML_predict:
	g++ -o swiftXML_predict $(CXXFLAGS) $(INC) swiftXML_predict.cpp swiftXML.cpp ../FastXML/fastXML.cpp ../Tools/c++/mat.cpp $(LIBFLAGS)

clean:
	rm -f swiftXML_train swiftXML_predict

