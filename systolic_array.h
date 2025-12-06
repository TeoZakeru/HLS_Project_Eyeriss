#pragma once
#include "configs.h"
#include "pe.h"

class EyerissF {
public:
    // The 2D array of PEs, matching the Python 'self.PEArray'
    PE PEArray[EyerissHeight][EyerissWidth];

    // Constructor
    EyerissF();

    // This is the main function logic, called by the HW wrapper
    void Conv2d(const data_t Picture[PIC_HEIGHT][PIC_WIDTH],
                const data_t FilterWeight[FLT_HEIGHT][FLT_WIDTH],
                int ImageNum, int FilterNum,
                data_t ConvedArray[OUT_HEIGHT][OUT_WIDTH]); // Output

private:
    // All private helper methods from SystolicArray.py
    void __InitPEs__();
    void __SetALLPEsState__(int State);
    void __SetPEsRunningState__(int PictureColumnLength, int FilterWeightColumnLength);
    void __SetALLPEImgNumAndFltNum__(int ImageNum, int FilterNum);

    void __DataDeliver__(const data_t Picture[PIC_HEIGHT][PIC_WIDTH],
                         const data_t FilterWeight[FLT_HEIGHT][FLT_WIDTH],
                         int ImageNum, int FilterNum,
                         int& PictureColumnLength, int& FilterWeightColumnLength);

    void __run__();

    void __PsumTransport__(data_t Result[OUT_HEIGHT][OUT_WIDTH],
                           int PictureColumnLength, int FilterWeightColumnLength);
                           
    void Relu2D(data_t array[OUT_HEIGHT][OUT_WIDTH]);
};

/**
 * @brief TOP-LEVEL FUNCTION FOR VITIS HLS SYNTHESIS.
 * This is the function you will set as the 'top' function in your HLS project.
 */
extern "C" {
void eyeriss_conv2d_hw(
    const data_t Picture[PIC_HEIGHT][PIC_WIDTH],
    const data_t FilterWeight[FLT_HEIGHT][FLT_WIDTH],
    data_t ConvedArray[OUT_HEIGHT][OUT_WIDTH]
);
}
