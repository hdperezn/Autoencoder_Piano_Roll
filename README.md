# Unet-Like for Piano Roll 
this repository contains:
* some important functions to process MIDI in order to use them in Deep learning Aplication.
* a neural network called "Unet-Like for Piano Roll" designed to rebuild piano roll arrays and generate a Mask with the "hits" of the input set

# Dependencies

*   fluidsynth
*   pyfluidsynth
*   pretty_midi
```python
!sudo apt install -y fluidsynth
!pip install --upgrade pyfluidsynth
!pip install pretty_midi
```

# Dataset:

The MIDI dataset was created with the tool  [Basic Pitch](https://basicpitch.spotify.com/) 
 from the audio stimuli presented in the [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/). 
It consists of 40 samples with a duration of 1 minute.

```python
path_midi = '/content/MIDI_functions/deap_midis'
data_dir = pathlib.Path(path_midi)
filenames = glob.glob(str(data_dir/'*.mid*'))
print('Number of files:', len(filenames))
```

# Functions: 

In this section, you could find:
* MIDI_functions: useful functions for visualizing the MIDI data and creating a Piano roll 
training dataset. 
* UNetLike_PianoRolls: Unet Like Model for piano roll representation


```python
from MIDI_functions.functions.MIDI_functions import cut_midi_secTrial
from MIDI_functions.functions.UNetLike_PianoRolls import UNet_Pianoroll
```

# Creating training Piano roll data set

For the experiment, a sampling rate from 16000 was used to sound the midi stimuli
and a frequency from 20 to generate the piano roll representation.
```python
_SAMPLING_RATE = 16000
fProll = 20
```

The "get_piano_roll" function from the [Pretty_MIDI](https://craffel.github.io/pretty-midi/) library was used to generate 
the piano roll representation, then every one-minute sample was cut into 10 segments, 
and the pitch data was cut under the 88 notes. Finally, some samples were deleted
because the original audio stimuli were shorter than 60 seconds

All of these resulting an array with dimensions (391, 128, 80) [N° samples, time, pitch] and 
the target mask is calculated by replacing with a one where the array was non-zero

```python
#train_test_data
Prolls_array = np.concatenate(np.asarray(Prolls_cut), axis = 0)
Prolls_array_mask = np.where(Prolls_array>0,1,0)
print('these are de dims of the piano roll array to use in the neural network: ', Prolls_array.shape)

```  

Of the 391 samples, 313 were selected for training, and the rest were used for testing.

```python
#train_test_data

X_train = Prolls_array[0:313,8:88,:,np.newaxis].astype('float32').transpose(0,2,1,3)
X_test = Prolls_array[313::,8:88,:, np.newaxis].astype('float32').transpose(0,2,1,3)

y_train = Prolls_array_mask[0:313,8:88,:,np.newaxis].astype('float32').transpose(0,2,1,3)
y_test = Prolls_array_mask[313::,8:88,:,np.newaxis].astype('float32').transpose(0,2,1,3)

X_dims = X_train.shape
print(X_dims)
```

# Load Model 

The Unet Like model for Piano Roll was inspired in the 
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
paper. The main difference is that the residual connections between the encoder and decoder 
were disconnected because this model was developed with the purpose to be used as a 
generative model.

The loss function to train the model was the Dice Coefficient, it was taken from the [GCPDS repository](https://github.com/UN-GCPDS/python-gcpds.image_segmentation.git).
After a selection of parameters, the best model was saved and can be loaded like these:


```python
img_size = X_dims[-3::]
num_classes = 1
loss = DiceCoefficient()
UNet_pianoroll = UNet_Pianoroll(img_size, loss, num_classes, epochs=250,batch_size=32,
               learning_rate=1e-3, droprate = 0.6)

UNet_pianoroll.get_model()
UNet_pianoroll.model.load_weights('/content/MIDI_functions/Unet_piano_rolls_models/UnetLike_PianoRoll_3encoding.h5')
UNet_pianoroll.model.summary()
```

You can run a demonstrative notebook in the UNetLike_pianorolls.ipyn

# bibliography 

* Colin Raffel and Daniel P. W. Ellis. Intuitive Analysis, Creation and Manipulation of MIDI Data with pretty_midi. In 15th International Conference on Music Information Retrieval Late Breaking and Demo Papers, 2014.
* KOELSTRA, Sander, et al. Deap: A database for emotion analysis; using physiological signals. IEEE transactions on affective computing, 2011, vol. 3, no 1, p. 18-31.
* “Image Segmentation with a U-Net-like Architecture”. Keras.Io, https://keras.io/examples/vision/oxford_pets_image_segmentation/. Consultado el 17 de marzo de 2023.
* RONNEBERGER, Olaf; FISCHER, Philipp; BROX, Thomas. U-net: Convolutional networks for biomedical image segmentation. En Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015. p. 234-241.
* 
