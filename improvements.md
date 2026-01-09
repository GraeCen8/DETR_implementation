# this is a list of *as many as possible* methods for improving model performance

### possible improvements 
- add Dice / Focal loss - better boundaries
- Better backbone - resnet encoder
- augment a **lot** - flips, scales, rotations
- weighted CE - rare classes matter more 
- TTA / post-processing - refine boundaries
these aim to get CE loss below 0.5 and mloU above 60%-70%
*this is production grade for most tasks*