#pragma once
#include "configs.h"

class PE {
public:
    // Member variables (public for easy access from EyerissF, matching Python)
    data_t FilterWeight[FLT_WIDTH];
    data_t ImageRow[PIC_WIDTH];
    data_t Psum[PE_PSUM_LENGTH];
    int PEState;
    int ImageNum;
    int FilterNum;

    // Constructor
    PE();

    // Methods from PE.py
    void SetPEState(int State);
    void SetFilterWeight(const data_t* InFilterWeight); // Pass pointer to row
    void SetImageRow(const data_t* InImageRow);     // Pass pointer to row
    void SetPEImgAndFlt(int ImgNum, int FltNum);
    void CountPsum();

private:
    // __SetPsum__ is GONE.
    void __Conv1d__(const data_t* ImgRow, const data_t* FltW, data_t* result);
    void __Conv__(data_t* result_array);
};
