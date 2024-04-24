# Leveraging Self-Supervised Vision Transformers for Segmentation-based Transfer Function Design
This is the official repository of the paper by [Dominik Engel](https://dominikengel.com), [Leon Sick](https://leonsick.github.io) and
[Timo Ropinski](https://viscom.uni-ulm.de/members/timo-ropinski/).

This repository contains
- Code to retrieve the feature volumes as detailed in the paper
- Evaluation Code
- Baselines

# Code for the Volume Renderer and GUI
The full project requires you to build [Inviwo](https://github.com/inviwo/inviwo) ([inviwo.org](https://inviwo.org)) together
with our [external Module](https://github.com/xeTaiz/inviwo-module-vittf).
Please visit the [repository of the external Module](https://github.com/xeTaiz/inviwo-module-vittf) for detailed instructions.

# Cite our Paper
```bibtex
@misc{engel2024vittf,
 abstract = {In volume rendering, transfer functions are used to classify structures of interest, and to assign optical properties such as color and opacity. They are commonly defined as 1D or 2D functions that map simple features to these optical properties. As the process of designing a transfer function is typically tedious and unintuitive, several approaches have been proposed for their interactive specification. In this paper, we present a novel method to define transfer functions for volume rendering by leveraging the feature extraction capabilities of self-supervised pre-trained vision transformers. To design a transfer function, users simply select the structures of interest in a slice viewer, and our method automatically selects similar structures based on the high-level features extracted by the neural network. Contrary to previous learning-based transfer function approaches, our method does not require training of models and allows for quick inference, enabling an interactive exploration of the volume data. Our approach reduces the amount of necessary annotations by interactively informing the user about the current classification, so they can focus on annotating the structures of interest that still require annotation. In practice, this allows users to design transfer functions within seconds, instead of minutes. We compare our method to existing learning-based approaches in terms of annotation and compute time, as well as with respect to segmentation accuracy. Our accompanying video showcases the interactivity and effectiveness of our method.},
 author = {Engel, Dominik and Sick, Leon and Ropinski, Timo},
 doi = {10.48550/arXiv.2309.01408},
 publisher = {arXiv},
 title = {Leveraging Self-Supervised Vision Transformers for Segmentation-based Transfer Function Design},
 year = {2023}
}
```
