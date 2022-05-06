import numpy as np
import pywt
import bokeh
import bokeh.plotting as bpl
from bokeh.models import ColorBar, BasicTicker, LinearColorMapper
import matplotlib
import fastmri
from fastmri.data import transforms as T
from PIL import Image

# Some functions taken from HW1 - MRI

## Standard basis element function for matrix space (ZERO-INDEXED)
def stdbel(n, i, j):
  E = np.zeros((n, n))
  E[i, j] = 1
  return E

## Try to do something like imagesc in MATLAB using Bokeh tools.
def imagesc(M, title=''):
  m, n = M.shape
  
  # 600 px should be good; calculate ph to try to get aspect ratio right
  pw = 600
  ph = round(1.0 * pw * m / n)
  h = bpl.figure(plot_width = pw, plot_height = ph, x_range=(0, 1.0*n),
                 y_range=(0, 1.0*m), toolbar_location='below',
                 title=title, match_aspect=True
                )
  
  minval = np.min(M)
  maxval = np.max(M)
  
  color_mapper = LinearColorMapper(palette="Greys256", low=minval, high=maxval)
  h.image(image=[M], x=0, y=0, dw=1.0*n, dh=1.0*m, color_mapper=color_mapper)
  
  color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                      label_standoff=12, border_line_color=None, location=(0, 0))
  
  h.add_layout(color_bar, 'right')
  

  bpl.show(h)
  return h

# Get a default slice object for a multilevel wavelet transform
# Used to abstract this annoying notation out of the transform...
def default_slices(levels, n):
  c = pywt.wavedec2(np.zeros((n, n)), 'db4', mode='periodization', level=levels)
  bye, slices = pywt.coeffs_to_array(c)
  return slices

# Wrapper for forward discrete wavelet transform
# Output data as a matrix (we don't care about tuple format)
def dwt(levels, sdom_data):
  c = pywt.wavedec2(sdom_data, 'db4', mode='periodization', level=levels)
  output, bye = pywt.coeffs_to_array(c)
  return output

# Wrapper for inverse discrete wavelet transform
# Expect wdom_data as a matrix (we don't care about tuple format)
def idwt(levels, wdom_data, slices=None):
  n = wdom_data.shape[0]
  if slices is None:
    slices = default_slices(levels, n)
  c = pywt.array_to_coeffs(wdom_data, slices, output_format='wavedec2')
  return pywt.waverec2(c, 'db4', mode='periodization')

def frob_error(X_reconstruct,X):
  num = np.linalg.norm(X_reconstruct-X, ord='fro')
  den = np.linalg.norm(X, ord='fro')
  return num/den

def k_encode(k,X):
  X_dwt = dwt(levels=3,sdom_data=X)
  X_sorted = np.sort(np.absolute(np.array(X_dwt)).flatten(),kind='quicksort')[::-1]
  k_threshold = X_sorted[k-1]
  X_dwt[np.absolute(X_dwt)<k_threshold] = 0
  X_encode = X_dwt
  return X_encode

def decode(X_encode):
  X_reconstruct = idwt(levels=3, wdom_data=X_encode)
  return X_reconstruct

def k_encode_q(k,X):
  X_encode = k_encode(k,X)
  return np.array(X_encode, dtype=np.float32)

def decode_q(X_encode):
  X_reconstruct = decode(X_encode)
  return np.array(X_reconstruct, dtype=np.float32)

def k_reconstruct(k,X_dwt):
  X_sorted = np.sort(np.absolute(np.array(X_dwt)).flatten(),kind='quicksort')[::-1]
  k_threshold = X_sorted[k-1]
  X_dwt[np.absolute(X_dwt)<k_threshold] = 0
  X_reconstruct = idwt(levels=3, wdom_data=X_dwt)
  return X_reconstruct

def evaluate_k(k,X):
  X_encode = k_encode(k,X)
  X_reconstruct = decode(X_encode)
  error = frob_error(X_reconstruct,X)
  return error

def resize_scan(X):
    # Raw image is a rectangle, crop to be square
    y_min = int((768-396)/2)
    y_max = int((768-396)/2+396)
    X_square = X[y_min:y_max,:]
    # Crop further to zone of interest to get a 256x256 image
    X_square_256 = X_square[50:50+256,70:70+256]
    return X_square_256

def to_jpeg(X,X_true):
    matplotlib.image.imsave('../images/X.jpg', X)
    image = Image.open('../images/X.jpg').convert('L')
    # convert image to numpy array
    X_jpeg = np.asarray(image.getdata()).reshape(image.size)
    X_jpeg = X_jpeg*np.max(X_true)/255
    return X_jpeg

def combine_all_coils(volume_kspace,slice_idx):
    slice_kspace = volume_kspace[slice_idx]
    slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
    slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
    slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image
    slice_image_rss = fastmri.rss(slice_image_abs, dim=0)
    return slice_image_rss.numpy()

def binary_search(low, high, accuracy, tolerance, X):
  guess = (high + low)//2
  if guess == low or guess == high:
    return guess
  guess_error = evaluate_k(guess, X)

  #print("Low: "+str(low)+",High: "+str(high))
  #print(guess_error)

  if guess_error - (1 - accuracy) < tolerance:
    return binary_search(low, guess-1, accuracy, tolerance, X)
  elif guess_error - (1 - accuracy) > tolerance:
    return binary_search(guess+1, high, accuracy, tolerance, X)
  else:
    return guess

# Quantized K-Wavelet compression
def quantize(X):
    Xmax = np.max(X)
    X_q = X*1/Xmax #normalize between 0-1
    X_q = np.array(X_q, dtype=np.float32)
    return X_q