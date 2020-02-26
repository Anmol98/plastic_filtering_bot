import tensorflow as tf
import sys
import os
import glob

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
score=[];
score1=[];
#def retval(i):
"""imdir = 'C:/Users/Amey Vijay Shimpi/Downloads/Plastic-Detection-Model-master (1)/Plastic-Detection-Model-master/Images'
ext = ['png', 'jpeg', 'jpg']
files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
images = [cv2.imread(file) for file in files]
"""
j=0;
for i in range(3):
    image_path="C:/Users/Amey Vijay Shimpi/Downloads/Plastic-Detection-Model-master (1)/Plastic-Detection-Model-master/Images/Plastic"+str(i+1)+".jpeg"
#image_path = "C:/Users/Amey Vijay Shimpi/Downloads/plastic1"+".jpeg"
#image_path = "C:/Users/Amey Vijay Shimpi/Desktop/parrot.png"


    if image_path:

    # Read the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line
        in tf.gfile.GFile("tf_files/retrained_labels.txt")]

    # Unpersists graph from file
        with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k :
           human_string = label_lines[node_id]
           score = predictions[0][node_id]
        print(predictions[0][4]);


            #print('%s (score = %.5f)' % (human_string, score))
"""imdir = 'C:/Users/Amey Vijay Shimpi/Downloads/Plastic-Detection-Model-master (1)/Plastic-Detection-Model-master/Images'
ext = ['png', 'jpeg', 'jpg']
files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
images = [cv2.imread(file) for file in files]
"""


"""for y in images:
    score[y]=retval(y)


score.sort()
"""
