# signature-verification
Signature forgery and fraud detection
We first extract cheques from a picture using a YOLO model. Clone the following repository and make the following changes

`git clone https://github.com/tensorflow/models.git`
Go into the models directory and clone the following repository.

`git clone https://github.com/rupeego/object_detection`

Now we need to train the dataset for our images, which can be achieved by cloning the images directory. This directory has the train test data while also further data to create tfrecords and find the authenticity of a signature. This code can be found in the Non-CNN directory. 

To get the images and pretrained model, go into the object_detection directory, remove the images folder and clone the following.

`git clone https://github.com/rupeego/images`

Coming back to the main object_detection folder, we can finally get the ssd_coconet file from the following command. 

`wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz`
or 
`wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz `

We check the protobuf version and install jupyter and matplotlib 
`sudo apt-get install protobuf-compiler python-pil python-lxml
sudo pip install jupyter
sudo pip install matplotlib`
And then:

# From tensorflow/models/
`protoc object_detection/protos/*.proto --python_out=.`
We now go back to the models directory and get the slim module. 
    `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`  and then `sudo python3 setup.py`
    
We are now ready to train the network, using the command. 
`python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco #use the config file of the model downloaded`


Alternatively a pretrained model can be be run directly from the given cheque_graph directory.
We now replace object_detection_tutorial.ipynb with object_detection_main.ipynb in the object_detection directory under the models directory.

The image to be detected need to be stored in the test_images directory, rename it as image3 for easy although the code can be changed to specify the image name. 

The image of the detected item is stored as cropped.jpg in the object_detection directory.

To train this model further, the latest of the checkpoints along with the data can be copied to the models/object_detection/training/ directory and training can be restarted. 

'''THE PATHS NEED TO BE RECONFIGURED DEPENDING ON THE LOCATION OF THE CLONED DIRECTORY'''.

The detected region then needs to be stored in the 
images/signature_extractor-master directory.

`python3 crop_images.py` specify the image name
The image should be copied into this directory from the object_detection directory. Consider changing names to avoid confusion.

`python3 sharpen-pil.py`

`python3 nearest-neighbour.py`

`python3 signature-extractor.py`

We store the output.png file in the Non\ CNN/for-verification folder.

Now, in Non CNN/Back-End/
`python3 kl.py`

Enter the name of the folder with genuine images. The name of the image to be verified and the threshhold divergence based on pre existing data. This value is preset to the maximum divergence in the given real signatures and cannot be less than this, it can be more given on how strict the parameter needs to be. 

The code will output the divergence along with the `accept` or `reject` depending on the kl distance of this image from the rest of the set which contains genuine signature samples.

When the number of signatures is large, the CNN code can be used by putting the genuine signatures in a named folder and running the signature to be verified through that particular trained model. The model is a simple CNN and outputs yes or no which signifies if the signature is geniune or forged. 
