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

`wget https://raw.githubusercontent.com/tensorflow/models/master/object_detection/samples/configs/ssd_mobilenet_v1_pets.config`


We check the protobuf version and install jupyter and matplotlib 
`sudo apt-get install protobuf-compiler python-pil python-lxml
sudo pip install jupyter
sudo pip install matplotlib
And then:

# From tensorflow/models/
protoc object_detection/protos/*.proto --python_out=.`
We now go back to the models directory and get the slim module. 
    `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`  and then `sudo python3 setup.py`
    
We are now ready to train the network, using the command. 
`python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco`

Alternatively a pretrained model can be be run directly from the given cheques_graph directory. 


