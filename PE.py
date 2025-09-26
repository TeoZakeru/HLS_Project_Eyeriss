import numpy as np
import configs


class PE:
    PEBuffer = configs.PEBuffer
    PEState = configs.ClockGate

    def __init__(self):
        self.SetFilterWeight((0, 0))
        self.SetImageRow((0, 0))

    def SetPEState(self, State):
        self.PEState = State

    def SetFilterWeight(self, FilterWeight):
        self.FilterWeight = FilterWeight

    def SetImageRow(self, ImageRow):
        self.ImageRow = ImageRow

    def SetPEImgAndFlt(self, ImageNum, FilterNum):
        self.ImageNum = ImageNum
        self.FilterNum = FilterNum

    def __SetPsum__(self, Psum):
        self.Psum = Psum

    def __Conv1d__(self, ImageRow, FilterWeight):
        result = list()
        for x in range(0, len(ImageRow) - 1 + len(FilterWeight)):
            y = x + len(FilterWeight)
            if y > len(ImageRow):
                break
            r = ImageRow[x:y] * FilterWeight
            result.append(r.sum())
        return np.array(result)

    def __Conv__(self):
        ImageRow = self.ImageRow
        FilterWeight = self.FilterWeight
        ImageNum = self.ImageNum
        FilterNum = self.FilterNum

        l = list()
        if FilterNum == 1 and ImageNum == 1:
            return self.__Conv1d__(ImageRow, FilterWeight)
        else:
            if FilterNum == 1:
                pics = np.hsplit(ImageRow, ImageNum)
                for x in pics:
                    l.append(self.__Conv1d__(x, FilterWeight))
                    result = np.hstack(np.array(l))
                return result
            if ImageNum == 1:
                FilterWeight = np.reshape(FilterWeight, (int(FilterWeight.size / FilterNum), FilterNum))
                flts = np.array(FilterWeight.T)
                for x in flts:
                    l.append(self.__Conv1d__(ImageRow, x))
                result = np.array(l)
                result = result.T
                result = np.reshape(result, (1, result.size))
                return result

    def CountPsum(self):
        if self.PEState == configs.ClockGate:
            self.__SetPsum__(configs.EmptyPsum)
        elif self.PEState == configs.Running:
            self.__SetPsum__(self.__Conv__())