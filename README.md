This is a Python 3 implementation of the Visual Information Fidelity (VIF) Image Quality Assessment (IQA) metric.

Requirements:
pyrtools >= 1.0.0

To compute VIF, use the following code snippet
```
# img_ref, img_dist are two images of the same size.
from vif_utils import vif
print(vif(img_ref, img_dist))
```
