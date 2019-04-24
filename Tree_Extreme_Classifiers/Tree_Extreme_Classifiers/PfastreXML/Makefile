SRC=../FastXML
CXXFLAGS=-std=c++11 -O3 -g
LIBFLAGS=-pthread
CXXINCPATH=../Tools/c++
INC=-I$(CXXINCPATH) -I$(SRC)

all: clean PfastreXML_train PfastreXML_predict

PfastreXML_train:
	$(CXX) -o PfastreXML_train $(CXXFLAGS) $(INC) PfastreXML_train.cpp $(SRC)/fastXML.cpp ../Tools/c++/mat.cpp $(LIBFLAGS)

PfastreXML_predict:
	$(CXX) -o PfastreXML_predict $(CXXFLAGS) $(INC) PfastreXML_predict.cpp $(SRC)/fastXML.cpp ../Tools/c++/mat.cpp $(LIBFLAGS)

clean:
	rm -f PfastreXML_train PfastreXML_predict
