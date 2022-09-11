#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Convolutional.h"


namespace ML {
    // --- Begin Student Code ---

    // Compute the convolution for the layer data
    void ConvolutionalLayer::computeNaive(const LayerData &dataIn) const {

        //std::cout << "\tDEBUG: conv layer" << std::endl;

        //read in convolution parameters
        const LayerParams convInputParams = this->getInputParams();
        const LayerParams convOutputParams = this->getOutputParams();
        const LayerParams convWeightParams = this->getWeightParams();
        const LayerParams convBiasParams = this->getBiasParams();

        //Read in convolution input data (NOTE: Array3D_fp32 = float***)
        Array3D_fp32 convInput = dataIn.getData<fp32***>();

        //assign the outData LayerData object's data pointer to a floating point array
        Array3D_fp32 convOutput = this->getOutputData().getData<fp32***>();
        
        //assign the convolutional kernel weights to a floating point array
        Array4D_fp32 kernelWeights = this->getWeightData().getData<fp32****>();
        Array1D_fp32 kernelBiases = this->getBiasData().getData<fp32*>();

        //Store Input dimensions
        const int H = convInputParams.dims[0];  //input height
        const int W = convInputParams.dims[1];  //input width
        const int C = convInputParams.dims[2];  //number of input fmaps

        //Store Output dimensions
        const int P = convOutputParams.dims[0]; //output height
        const int Q = convOutputParams.dims[1]; //output width

        //Store Kernel dimensions
        const int R = convWeightParams.dims[0];   //filter height
        const int S = convWeightParams.dims[1];   //filter width
        const int M = convWeightParams.dims[3];   //number of filters/output fmaps

        /*
        std::cout << "\t DEBUG: M = " << M << std::endl;
        std::cout << "\t DEBUG: C = " << C << std::endl;
        std::cout << "\t DEBUG: H = " << H << std::endl;
        std::cout << "\t DEBUG: W = " << W << std::endl;
        std::cout << "\t DEBUG: R = " << R << std::endl;
        std::cout << "\t DEBUG: S = " << S << std::endl;
        std::cout << "\t DEBUG: P = " << P << std::endl;
        std::cout << "\t DEBUG: Q = " << Q << std::endl;
        */
        //shift kernel position down 1 row
        for (int p = 0; p < P; p++){
            //shift kernel location over 1 column
            for (int q = 0; q < Q; q++){

                //sum up the product of one of the M convolution kernels with all C input channels
                for (int m = 0; m < M; m++){
                    
                    //initialize the running sum of kernel products over all C input channels
                    float sum = 0.0;

                    //flip & shift using convolution kernel on all C input channels
                    for (int c = 0; c < C; c++){
                        for (int r = 0; r < R; r++){
                            for (int s = 0; s < S; s++){
                                sum += convInput[p + r][q + s][c] * kernelWeights[r][s][c][m];
                            }
                        }
                    }

                    //assign one pixel of convolution output
                    convOutput[p][q][m] = sum + kernelBiases[m];

                    //ReLU
                    if (convOutput[p][q][m] < 0.0){
                        convOutput[p][q][m] = 0.0;
                    }
                }
            }
        }

        //std::cout << "\t\t...done computing" << std::endl;


    }


    // Compute the convolution using threads
    void ConvolutionalLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using a tiled approach
    void ConvolutionalLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using SIMD
    void ConvolutionalLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }
};