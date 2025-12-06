#pragma once

// --- From configs.py ---
const int EyerissWidth = 14;
const int EyerissHeight = 12;

// PE State
const int ClockGate = 0;
const int Running = 1;
const int EmptyPsum = 0;

// --- From main.py & logic ---
// Input picture dimensions
const int PIC_HEIGHT = 18; // len(Picture)
const int PIC_WIDTH = 18;  // len(Picture[0])

// Input filter dimensions
const int FLT_HEIGHT = 5; // len(FilterWeight)
const int FLT_WIDTH = 5;  // len(FilterWeight[0])

// --- Deduced from computation ---
// Output of 1D conv in PE
// Python: len(ImageRow) - len(FilterWeight) + 1 = PIC_WIDTH - FLT_WIDTH + 1
const int PE_PSUM_LENGTH = 14; // 18 - 5 + 1 = 14

// Output of final __PsumTransport__
// Python: PictureColumnLength + 1 - FilterWeightColumnLength = PIC_HEIGHT + 1 - FLT_HEIGHT
const int OUT_HEIGHT = 14; // 18 + 1 - 5 = 14
// Number of cols = PE_PSUM_LENGTH
const int OUT_WIDTH = 14;

// Data type for calculations
typedef int data_t;
