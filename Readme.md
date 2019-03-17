# Deep Learning - Lyrics Generator 

A Deep Learning architecture for lyrics generation based on a specific musical genre. Different languages has absolutily different performances on the generated texts and demands a different model setting to have a resonable result on a new language.

## Data

All lyrics (text-data) used to train the model was obtained using this Crawler <https://github.com/gabrielgoncalves95/vagalumeCrawler>.

## Environment and Libraries

### Python 3.6.5+
An Anaconda Environment is highly recommended because of its improvements on tensorflow performance, but any compatible environment will be able tu run the scripts.

*  Keras 2.2.2+

For Anaconda environment:
```
    conda install -c anaconda keras
```
For other environments use pip:
```
    pip install Keras==2.2.2
```
In case you want the most recent version:
```
    pip install Keras
```

* Tensorflow 1.10.0+ (Used as keras backend)

To use cpu as model processor:
```
    pip install tensorflow==1.10.0
```
To use gpu as model processor:
```
    pip install tensorflow-gpu==1.10.0
```
In case you want the most recent version:
```
    pip install tensorflow
```
```
    pip install tensorflow-gpu
```
To use the GPU Tensorflow, other hardware and software requirements must be satisfied (GPU drivers and CUDA settings), there's a tutorial to all of them on tensorflow's official site: <https://www.tensorflow.org/install/gpu>.

* Numpy

For Anaconda environment:
```
    conda install -c anaconda numpy
```
For other environments use pip:
```
    pip install numpy==1.14.5
```
In case you want the most recent version:
```
    pip install numpy
```


## Training

To configure the main characteristics of the network, like type and architecture, you can change the file 'modelconfig.py'. Many other settings of training and prediction can be changed in the other files.

With all of the environment requirements satisfied, to start a training:

```
    python training.py path-to-file-textfile.txt
```
After each epoch (if the current epoch got a better result than all last results) a model will be saved inside the folder "Models".

## Generating Lyrics (text)
To generate text with the saved models:
```
    python predict.py path-to-file-textfile.txt path-to-modelfile.hdf5 number-of-chars-to-predict
```
