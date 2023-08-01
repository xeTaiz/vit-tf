# Name: NumpyVolumeSource

import inviwopy as ivw
from inviwopy.properties import FileProperty, BoolProperty, StringProperty
from inviwopy.data       import VolumeOutport, Volume
from inviwopy.glm        import dvec2

import numpy as np
from pathlib import Path



class NumpyVolumeSource(ivw.Processor):
    def __init__(self, id, name):
        ivw.Processor.__init__(self, id, name)
        self.outport = VolumeOutport("outport")
        self.addOutport(self.outport, owner=False)

        self.outport_id = StringProperty('outport_id', 'Outport ID', 'outport')
        self.outport_id.onChange(self.updateOutportId)
        self.vol_path = FileProperty('filename', 'Volume path')
        self.normalize = BoolProperty('normalize', 'Fix Data Range to [0,1]', True)
        self.outputAs = ivw.properties.OptionPropertyString("outputAs", "Output As", [
            ivw.properties.StringOption("uint8", 'UINT8', 'u8'),
            ivw.properties.StringOption("uint16", 'UINT16', 'u16'),
            ivw.properties.StringOption("float16", 'FLOAT16', 'f16'),
            ivw.properties.StringOption("float32", 'FLOAT32', 'f32'),
            ivw.properties.StringOption('asis', "AS IS", 'asis')
        ])
        self.flipX = BoolProperty('flipX', 'Flip X', False)
        self.flipY = BoolProperty('flipY', 'Flip Y', False)
        self.flipZ = BoolProperty('flipZ', 'Flip Z', False)
        self.addProperty(self.vol_path)
        self.addProperty(self.normalize)
        self.addProperty(self.outputAs)
        self.addProperty(self.outport_id)
        self.addProperty(self.flipX)
        self.addProperty(self.flipY)
        self.addProperty(self.flipZ)

        self.vol = None

    def updateOutportId(self):
        self.removeOutport(self.outport)
        self.outport = VolumeOutport(self.outport_id.value)
        self.addOutport(self.outport, owner=False)

    @staticmethod
    def processorInfo():
        return ivw.ProcessorInfo(
            classIdentifier = "org.inviwo.numpyvolumesource",
            displayName = "Numpy Volume Source",
            category = "Python",
            codeState = ivw.CodeState.Stable,
            tags = ivw.Tags.PY
        )

    def getProcessorInfo(self):
        return NumpyVolumeSource.processorInfo()

    def initializeResources(self):
        path = Path(self.vol_path.value)
        if path.exists() and path.suffix == '.npy':
            self.vol = np.load(path, allow_pickle=True)
            print(f'Loaded volume ({tuple(self.vol.shape)}) ({self.vol.dtype}) in [{self.vol.min()}, {self.vol.max()}]')
            if self.normalize.value:
                self.vol = self.vol.astype(np.float32)
                self.vol = (self.vol - self.vol.min()) / (self.vol.max() - self.vol.min())
            if self.outputAs.value == 'u8':
                self.vol = (255.0 * self.vol).astype(np.uint8)
            elif self.outputAs.value == 'u16':
                self.vol = (65535.0 * self.vol).astype(np.uint16)
            elif self.outputAs.value == 'f16':
                self.vol = self.vol.astype(np.float16)
            elif self.outputAs.value == 'f32':
                self.vol = self.vol.astype(np.float32)
            elif self.outputAs.value == 'asis':
                pass
            else:
                raise Exception(f'Invalid output format: {self.outputAs.value}')
            print(f'Contiguity:  C: {self.vol.flags["C_CONTIGUOUS"]}     F: {self.vol.flags["F_CONTIGUOUS"]}')

    def process(self):
        if self.vol is not None:
            print(f'NumpyVolumeSource: Setting Outport: {self.vol.shape} ({self.vol.dtype})')
            vol = self.vol
            if self.flipX.value:
                vol = np.flip(vol, axis=-3)
            if self.flipY.value:
                vol = np.flip(vol, axis=-2)
            if self.flipZ.value:
                vol = np.flip(vol, axis=-1)

            if vol.ndim == 4:
                volume = Volume(np.ascontiguousarray(np.transpose(vol, (2,1,0,3)).reshape(vol.shape)))
            elif vol.ndim == 3:
                volume = Volume(np.asfortranarray(vol))
            else:
                raise Exception(f'Invalid volume dimension: {vol.ndim}')
            print('Loaded volume min/max: ', vol.min(), vol.max())
            volume.dataMap.dataRange = dvec2(vol.min(), vol.max())
            volume.dataMap.valueRange= dvec2(vol.min(), vol.max())
            # volume.interpolation = ivw.data.InterpolationType.Nearest
            self.outport.setData(volume)
