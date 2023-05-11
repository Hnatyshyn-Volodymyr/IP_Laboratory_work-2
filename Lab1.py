import numpy as np
import matplotlib.pyplot as pplot

from PIL import Image
import cv2
import imageio

from Neuron import *

def readHandImage(isOwn, isBinary, isGrey):
    dataOwnTestList = []

    if(isBinary):
        for imageFile in testImageFiles:
            image = cv2.imread("./files/own/" + imageFile)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
            bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
            out_gray = cv2.divide(image, bg, scale = 255)
            out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1] 

            cv2.imwrite("./files/binary/" + imageFile, out_binary)

    if(isGrey):
        for imageFile in testImageFiles:
            image = cv2.imread("./files/own/" + imageFile)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
            bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
            out_gray = cv2.divide(image, bg, scale = 255)

            cv2.imwrite("./files/gray/" + imageFile, out_gray)

    for imageFile in testImageFiles:
        if(isOwn):
            image = Image.open("./files/own/" + imageFile)
        if(isBinary):
            image = Image.open("./files/binary/" + imageFile)
        if(isGrey):
            image = Image.open("./files/gray/" + imageFile)
        new_image = image.resize((28, 28))

        img_arr = numpy.array(new_image);
        if(isOwn):
            img_arr = img_arr[:,:,0]
        img_data = ((255.0 - img_arr) / 255.0 * 0.99) + 0.01
        img_data = numpy.insert(img_data, 0, int(imageFile[3]))
        dataOwnTestList.append(img_data)

    return dataOwnTestList

testImageFiles = ["img0_0.png","img0_1.png",
                  "img1_0.png","img1_1.png",
                  "img2_0.png",
                  "img3_0.png","img3_1.png",
                  "img4_0.png","img4_1.png",
                  "img5_0.png","img5_1.png",
                  "img6_0.png",
                  "img7_0.png",
                  "img8_0.png",
                  "img9_0.png","img9_1.png"]

test_data_file = open("./mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
training_data_file = open("./mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

input_nodes = 28 * 28
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
epochs = 50

neuron = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

print("Start training")

for e in range(epochs):
    print("\nEpoch: ", e)

    for record in training_data_list:
        image = record.split(',')
        inputs = (numpy.asfarray(image[1:]) / 255.0 * 0.99) + 0.01

        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(image[0])] = 0.99

        neuron.train(inputs, targets)
       
    pass

print("\nEnd training")

scorecard = []

print("Start evaluation")

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    print("comparation:")
    print(correct_label, "correct label")
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = neuron.query(inputs)
    label = numpy.argmax(outputs)
    print(label, "network answer\n")
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)


def evalNeurons(data, msg, rotate):
    scorecard = []

    print("\n\nStart evaluation by "+msg+" images")
    i = 0

    for record in data:
        correct_label = int(record[0])
        inputs = numpy.asfarray(record[1:])
        if(rotate):
            inputs_plus_img = scipy.ndimage.rotate(inputs.reshape(28,28), 10, cval=0.01, reshape=False).flatten()
            inputs_minus_img = scipy.ndimage.rotate(inputs.reshape(28,28), -10, cval=0.01, reshape=False).flatten()
       
        outputs = neuron.query(inputs)
        if(rotate):
            outputs = numpy.append(outputs, neuron.query(inputs_plus_img))
            outputs = numpy.append(outputs, neuron.query(inputs_minus_img))
        label = numpy.argmax(outputs) % 10;
        print("file: ", testImageFiles[i], ";\texpected: ", correct_label, ";\toutput: ",label, ";\tResult: ",label == correct_label)
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        i+=1
        pass

    print("\nEnd evaluation by "+msg+" images")

    scorecard_array = numpy.asarray(scorecard)
    print("performance by "+msg+" images = ", scorecard_array.sum() / scorecard_array.size)


evalNeurons(readHandImage(1,0,0), "Origin without rotate", 0)
evalNeurons(readHandImage(1,0,0), "Origin with rotate", 1)

print("\n\n\n")

evalNeurons(readHandImage(0,1,0), "Binary without rotate", 0)
evalNeurons(readHandImage(0,1,0), "Binary with rotate", 1)

print("\n\n\n")

evalNeurons(readHandImage(0,0,1), "Grey without rotate", 0)
evalNeurons(readHandImage(0,0,1), "Grey with rotate", 1)

