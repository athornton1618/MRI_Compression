# Deep MRI Compression
ELEN E6876 Research Project - Columbia University Spring 2022

<p align="center">
  <img src=https://github.com/athornton1618/MRI_Compression/blob/main/images/slice_visualization.png width="500">
<p/>

## Contributors
1. Alex Thornton     (apt2141@columbia.edu)	

## Abstract
The purpose of this research is to create a model
which can efficiently store slices of MRI scans. Efficiency is ex-
plored across two optimization domains; minimization of genera-
tional loss through recursive compression, and compressed space
minimization. Some techniques and frameworks explored include
k-wavelet approximation, and various auto-encoder frameworks
for neural networks. Training and test data was provided by
FastMRI, a joint venture between NYU and Facebook (Meta).


Index Terms—Wavelet, Medical Imaging, Compression, Sparse
Auto Encoder, Convolutional Neural Network

## Use Case

* JPEG compression is optimized for general images
* But what about images that share common features?
* Various deep learning and signal processing techniques are explored for new MRI compression schemes 

## Dataset
* Approved use of [FastMRI](https://fastmri.med.nyu.edu/) dataset from Facebook (Meta) and NYU
* Only using subset of the multicoil_test dataset ~34.2GB zipped
* 1 brain for test (16 slices), 26 brains for evaluation (14 - 16 slices each)
* Open source [software suite](https://github.com/facebookresearch/fastMRI) for slicing and manipulating data

## Generational Loss

### JPEG Generational Loss
* Recursive JPEG compressions quickly erodes quality (photocopier effect)
* After only a few iterations, JPEG compression succumbs to noise corruption

<p align="center">
  <img src=https://github.com/athornton1618/MRI_Compression/blob/main/images/generation_loss_jpeg.jpg width="350" >
<p/>

<p align="center">
  <img src=https://github.com/athornton1618/MRI_Compression/blob/main/images/jpeg_recursion.jpg width="350">
<p/>

### Quantized K-Wavelet Generational Loss

* Wavelet basis is very resilient to recursive operations
* Subject k-wavelet coefficients to float32 quantization
* Results in quality loss in only first compression!
* Compression size of 14.4 KB

<p align="center">
  <img src=https://github.com/athornton1618/MRI_Compression/blob/main/images/generation_loss_wavelet.jpg width="350">
<p/>

<p align="center">
  <img src=https://github.com/athornton1618/MRI_Compression/blob/main/images/quantized_k_wavelet_recursion.jpg width="350">
<p/>

<p align="center">
  <img src=https://github.com/athornton1618/MRI_Compression/blob/main/images/generational_loss_plot.jpg width="350">
<p/>

## Deep Learning
* Can compress further with sparse auto-encoder via convolutional neural network framework
* Captures image features well, but very lossy

<p align="center">
  <img src=https://github.com/athornton1618/MRI_Compression/blob/main/images/cnn_sae_error_plot.jpg width="350">
<p/>

<p align="center">
  <img src=https://github.com/athornton1618/MRI_Compression/blob/main/images/cnn_sae_17424.jpg width="350">
<p/>

<p align="center">
  <img src=https://github.com/athornton1618/MRI_Compression/blob/main/images/cnn_sae_363.jpg width="350">
<p/>

## Learn More
* Read the full [paper](https://github.com/athornton1618/MRI_Compression/tree/main/documentation/MRI_Compression.pdf)
* Check out the [presentation](https://github.com/athornton1618/MRI_Compression/tree/main/documentation/apt2141_ELEN6876_presentation.pptx)

## Try It Yourself
1. Clone this repository
```
git clone www.github.com/athornton1618/MRI_Compression.git
cd MRI_Compression
pip install -r requirements.txt
```
3. Obtain permission for [FastMRI](https://fastmri.med.nyu.edu/) dataset
4. Download multicoil_test dataset with following command:
```
curl -C - "https://fastmri-dataset.s3.amazonaws.com/brain_multicoil_test.tar.gz?AWSAccessKeyId=YOUR TOKEN" --output brain_multicoil_test.tar.gz
mkdir MRI_Compression/data
```
3. Unzip multicoil_test
4. Move files to MRI_Compression/data/multicoil_test
5. You can run any of the scripts locally on your machine from this point, except sparse_autoencoder_cnn_colab.ipynb
6. Run this python command to build the dataset for Colab:
```
cd utils
python make_dataset.py
```
7. Open sparse_autoencoder_cnn_colab.ipynb in Google Colab
8. Under the 'Files' tab in Colab, make directory /data, and upload the following from MRI_Compression/data:
    * train_set_image.json
    * train_set_wavelet.json
    * test_set_image.json
    * test_set_wavelet.json
9. Run sparse_autoencoder_cnn_colab.ipynb

## One More Thing
This project marks the completion of my M.Sc. at Columbia University. 

I'd like to thank my Mom, Dad, sister Carolyn, brother Matt, and Murphy 🐕! 

Roar Lion Roar 👑!

<p align="center">
  <img src=https://github.com/athornton1618/MRI_Compression/blob/main/images/roar2022.JPG width="300">
<p/>

## Thank You
Copyright (c) 2022 Alex Thornton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
