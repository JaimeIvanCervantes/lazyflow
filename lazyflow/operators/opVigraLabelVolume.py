from lazyflow.graph import Operator, InputSlot, OutputSlot

import numpy
import vigra
import logging
from lazyflow.roi import extendSlice

logger = logging.getLogger(__name__)

class OpVigraLabelVolume(Operator):
    """
    Operator that simply wraps vigra's labelVolume function.
    """
    name = "OpVigraLabelVolume"
    category = "Vigra"
    
    Input = InputSlot() 
    BackgroundValue = InputSlot(optional=True)
    
    Output = OutputSlot()
    
    def setupOutputs(self):
        inputShape = self.Input.meta.shape

        # Must have at most 1 time slice
        timeIndex = self.Input.meta.axistags.index('t')
        assert timeIndex == len(inputShape) or inputShape[timeIndex] == 1
        
        # Must have at most 1 channel
        channelIndex = self.Input.meta.axistags.channelIndex
        assert channelIndex == len(inputShape) or inputShape[channelIndex] == 1

        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.dtype = numpy.uint32
        
    def execute(self, slot, subindex, roi, destination):
        assert slot == self.Output
        
        inputData = self.Input(roi.start, roi.stop).wait()
        inputData = inputData.view(vigra.VigraArray)
        inputData.axistags = self.Input.meta.axistags

        # Drop the time axis, which vigra.labelVolume doesn't remove automatically
        axiskeys = [tag.key for tag in inputData.axistags]        
        if 't' in axiskeys:
            inputData = inputData.bindAxis('t', 0)

        # Drop the channel axis, too.
        if 'c' in axiskeys:
            inputData = inputData.bindAxis('c', 0)

        inputData = inputData.view(numpy.ndarray)

        if self.BackgroundValue.ready():
            bg = self.BackgroundValue.value
            if isinstance( bg, numpy.ndarray ):
                # If background value was given as a 1-element array, extract it.
                assert bg.size == 1
                bg = bg.squeeze()[()]
            if isinstance( bg, numpy.float ):
                bg = float(bg)
            else:
                bg = int(bg)
            result =  vigra.analysis.labelVolumeWithBackground(inputData, background_value=bg).view(numpy.ndarray)
        else:
            result =  vigra.analysis.labelVolumeWithBackground(inputData).view(numpy.ndarray)
        return result.reshape(destination.shape)

    def propagateDirty(self, inputSlot, subindex, roi):
        if inputSlot == self.Input:
            # Extend the region by 1 pixel
            dirtyRoi = extendSlice(roi.start, roi.stop, self.Input.meta.shape, 1,1)
            self.Output.setDirty(dirtyRoi[0], dirtyRoi[1])
        elif inputSlot == self.BackgroundValue:
            self.Output.setDirty( slice(None) )

