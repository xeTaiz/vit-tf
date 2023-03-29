import inviwopy as ivw
import numpy as np
from pathlib import Path

from inviwopy.data import Volume, VolumeInport
from inviwopy.properties import ButtonProperty, StringProperty, DirectoryProperty, FileProperty, BoolProperty
from inviwopy.glm import vec4

class volume_numpy_export(ivw.Processor):
    def __init__(self, id, name):
        super().__init__(id, name)
        self.vol_in = VolumeInport("inport")
        self.addInport(self.vol_in)

        self.save_path  = FileProperty("savepath",  "Save Path", "")
        self.override_datarange = BoolProperty("overriderange", "Override Data Range", False)
        self.custom_datarange = ivw.properties.FloatVec4Property("customrange", "Custom Data Range", vec4(4095), vec4(0), vec4(65535))
        self.addProperty(self.save_path)
        self.addProperty(self.override_datarange)
        self.addProperty(self.custom_datarange)

        self.btn = ButtonProperty("btn", "Save Volume", self.save_volume)
        self.addProperty(self.btn)

    def save_volume(self):
        p = Path(self.save_path.value)
        if p.parent.exists() and self.vol_in.hasData():
            vol = self.vol_in.getData().data.astype(np.float32)
            print('Mins: ', vol.min(axis=(0,1,2)))
            print('Maxes: ', vol.max(axis=(0,1,2)))
            if self.override_datarange.value:
                print('Using custom data range')
                div = self.custom_datarange.value.array[None,None,None,:vol.shape[-1]]
                vol = (vol / div).clip(0,1)
            else:
                print('Min-Max normalizing each channel')
                vol = (vol - vol.min(axis=(0,1,2))) / (vol.max(axis=(0,1,2)) - vol.min(axis=(0,1,2)))
            np.save(p, vol)
            print(f'Saved volume {vol.shape} {vol.dtype} in [{vol.min(axis=(0,1,2))}, {vol.max(axis=(0,1,2))}] to {p}.')
        else: print('Invalid save directory')

    def initializeResources(self):
        pass

    def process(self):
        pass

    @staticmethod
    def processorInfo():
        return ivw.ProcessorInfo(
    		classIdentifier = "org.inviwo.volume_numpy_export",
    		displayName = "Volume Numpy Export",
    		category = "Python",
    		codeState = ivw.CodeState.Stable,
    		tags = ivw.Tags.PY
        )

    def getProcessorInfo(self):
        return volume_numpy_export.processorInfo()
