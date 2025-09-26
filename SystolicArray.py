import numpy as np
import configs
from PE import PE

class EyerissF:
    GlobalBuffer = configs.SRAMSize
    EyerissWidth = configs.EyerissWidth
    EyerissHeight = configs.EyerissHeight

    def __init__(self):
        self.__InitPEs__()

    def Conv2d(self, Picture, FilterWeight, ImageNum, FilterNum):
        PictureColumnLength, FilterWeightColumnLength = self.__DataDeliver__(Picture, FilterWeight, ImageNum, FilterNum)
        self.__run__()
        ConvedArray = self.__PsumTransport__(PictureColumnLength, FilterWeightColumnLength)
        ReluedConvedArray = self.Relu(ConvedArray)
        self.__SetALLPEsState__(configs.ClockGate)

        return ReluedConvedArray

    def __InitPEs__(self, PEsWidth=configs.EyerissWidth, PEsHeight=configs.EyerissHeight):
        self.PEArray = list()
        for x in range(0, PEsHeight):
            self.PEArray.append(list())
            for y in range(0, PEsWidth):
                self.PEArray[x].append(PE())

    def __SetALLPEsState__(self, State):
        assert State == configs.ClockGate or State == configs.Running
        for ColumnELement in range(0, EyerissF.EyerissHeight):
            for RowElement in range(0, EyerissF.EyerissWidth):
                self.PEArray[ColumnELement][RowElement].SetPEState(State)

    def __SetPEsRunningState__(self, PictureColumnLength, FilterWeightColumnLength):
        assert FilterWeightColumnLength <= PictureColumnLength
        assert FilterWeightColumnLength <= EyerissF.EyerissHeight
        assert PictureColumnLength <= EyerissF.EyerissHeight + EyerissF.EyerissWidth - 1

        for ColumnELement in range(0, FilterWeightColumnLength):
            for RowElement in range(0, PictureColumnLength + 1 - FilterWeightColumnLength):
                try:
                    self.PEArray[ColumnELement][RowElement].SetPEState(configs.Running)
                except:
                    pass

    def __SetALLPEImgNumAndFltNum__(self, ImageNum, FilterNum):
        for ColumnELement in range(0, EyerissF.EyerissHeight):
            for RowElement in range(0, EyerissF.EyerissWidth):
                self.PEArray[ColumnELement][RowElement].SetPEImgAndFlt(ImageNum, FilterNum)

    def __DataDeliver__(self, Picture, FilterWeight, ImageNum, FilterNum):
        assert len(FilterWeight) <= self.EyerissHeight
        assert len(Picture) <= len(FilterWeight) + self.EyerissWidth - 1

        PictureColumnLength = len(Picture)
        FilterWeightColumnLength = len(FilterWeight)

        self.__SetALLPEImgNumAndFltNum__(ImageNum, FilterNum)
        self.__SetPEsRunningState__(PictureColumnLength, FilterWeightColumnLength)
        for ColumnELement in range(0, len(FilterWeight)):
            for RowElement in range(0, self.EyerissWidth):
                self.PEArray[ColumnELement][RowElement].SetFilterWeight(FilterWeight[ColumnELement])
        for ColumnELement in range(0, len(Picture)):
            DeliverinitR = 0
            DeliverinitH = ColumnELement
            for c in range(0, ColumnELement + 1):
                try:
                    self.PEArray[DeliverinitH][DeliverinitR].SetImageRow(Picture[ColumnELement])
                except:
                    pass
                DeliverinitR = DeliverinitR + 1
                DeliverinitH = DeliverinitH - 1

        return PictureColumnLength, FilterWeightColumnLength

    def __run__(self):
        for x in range(0, configs.EyerissHeight):
            for y in range(0, configs.EyerissWidth):
                if self.PEArray[x][y].PEState == configs.Running:
                    self.PEArray[x][y].CountPsum()

    def __PsumTransport__(self, PictureColumnLength, FilterWeightColumnLength):
        line = list()
        result = list()
        for RowElement in range(0, PictureColumnLength + 1 - FilterWeightColumnLength):
            line.clear()
            for ColumnElement in range(0, FilterWeightColumnLength).__reversed__():
                line.append(self.PEArray[ColumnElement][RowElement].Psum)
            result.append(np.sum(line, axis=0, dtype=int))
        if result == []:
            return
        return np.vstack(result)

    def __ShowPEState__(self, x, y):
        print("PE is : ", x, ",", y)

        if self.PEArray[x][y].PEState == configs.Running:
            print("PEState : Running")
        else:
            print("PEState : ClockGate")

        print("FilterWeight :", self.PEArray[x][y].FilterWeight)
        print("ImageRow :", self.PEArray[x][y].ImageRow)
        
    def ReluArray(self, array):
        assert type(array) == type(list())
        return [self.Relu(x)  for x in array]
    
    def Relu(self, array):
        array[array < 0] = 0
        return array