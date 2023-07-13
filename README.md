# iv
Intravenous Image Viewer and upscaler experiments

### TODO:

- [ ] Implement ICC color profiles.
   * see: https://learn.microsoft.com/en-us/windows/win32/wcs/alphabetical-list-of-all-wcsfunctions
   * ICC profile is spawn across APP2 tags of JPG images
   * see: https://github.com/drewnoakes/metadata-extractor/issues/65

- [ ] Implement async reading of next image from the folder and keyboard left/right space navigation

- [ ] Implement going into subfolders (maybe)

- [ ] Implement upscaling DNN neural net
    Using input image itself as a 3x3 -> 2x2 training set:

    1 Downsample image by 2x

    2 for each 3x3 pixel surroundings connect 9 values to a single dense layer with the
      activation function (e.g. tanh) and bias and connect outputs of dense
      layer to 2x2 = 4 RGB values of upscaled image.

    3 Train using all 3x3 boxes on downscaled image mapping them to 2x2 original image
      calculating loss and back propagation.

    4 Try for several epochs and experiment with width of dense layer,
      different activation functions.

    5 Do inference.

    6 Optimize using OpenCL where available and AVX

- [ ] UX improvements: Upscale -> Save to filename.2x.ext, Side by Side zoom split screen.

- [ ] Users feedback.

