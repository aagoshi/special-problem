'''
Use Keras API from Tensorflow to create the CNN model

Will use Youngmin(2019) CNN architecture
'''

import random
import os
import numpy as np
import cv2 
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras import callbacks
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

def view_image(img):
    cv2.imshow("test",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def menu4():
    print(
    '''
    ---------------------MENU----------------------------
    [1] wo
    [2] wl
    [3] wh
    [4] so
    [5] sl
    [6] sh
    -----------------------------------------------------  
    ''')
    choice = input("choose from menu: ")
    if choice == "1" : choice = "wo"
    elif choice == "2" : choice = "wl"
    elif choice == "3" : choice = "wh"
    elif choice == "4" : choice = "so"
    elif choice == "5": choice = "sl"
    elif choice == "6": choice = "sh"
    else: print("wrong input...")
    return choice

def main_menu():
    print(
    '''
    ---------------------MENU----------------------------
    [1] Filipino vs nonFilipino classification
    [2] Author classification
    -----------------------------------------------------  
    ''')
    choice = input("choose which to classify: ")
    return choice

def load_general_datasets(format):
    #Filipino vs non-Filipino dataset
    filipino_dataset_path = "dataset/filipino/original/" + format
    nonfilipino_dataset_path = "dataset/nonfilipino/original/" + format

    #Load the filipino and nonFilipino datasets into an array each
    filipino_dataset = []
    nonfilipino_dataset = []
    for file in os.listdir(filipino_dataset_path):
        image = cv2.imread(filipino_dataset_path + "/" + file)
        filipino_dataset.append(image)
    for file in os.listdir(nonfilipino_dataset_path):
        image = cv2.imread(nonfilipino_dataset_path + "/" + file)
        nonfilipino_dataset.append(image)
    
    #shuffle the contents of each dataset; authors do not matter
    random.shuffle(filipino_dataset)
    random.shuffle(nonfilipino_dataset)

    #smaller size dataset will be maximum length when dividing
    len_of_dataset = min(len(filipino_dataset),len(nonfilipino_dataset)) 

    #set train-validate-test size  at 80:10:10 ratio
    train_size = int(len_of_dataset *0.8) 
    val_size = int(len_of_dataset *0.1)
    test_size = int(len_of_dataset *0.1) 
    print("Ratio of images: train:"+str(train_size)+ " validation: " + str(val_size) + " test_size: "+ str(test_size))
    
    #0 represents filipino class, 1 represents non-filipino class
    #prepare the training image dataset and its corresponding labels
    train = []
    for k in range(0,train_size):
        train.append([filipino_dataset[k],0])
        train.append([nonfilipino_dataset[k],1])
    random.shuffle(train)
    train_dataset = []
    train_label = []
    for item, label in train:
        train_dataset.append(item)
        train_label.append(label)

    #prepare the validation image dataset and its corresponding labels
    val_dataset = []
    val_label = []
    upper_val = train_size + val_size
    for k in range(train_size, upper_val):
        val_dataset.append(filipino_dataset[k])
        val_label.append(0)
    for k in range(train_size,upper_val):
        val_dataset.append(nonfilipino_dataset[k])
        val_label.append(1)
    
    #prepare the test image dataset and its corresponding labels
    test_dataset = []
    test_label = []
    upper_test = upper_val + test_size
    for k in range(upper_val, upper_test):
        test_dataset.append(filipino_dataset[k])
        test_label.append(0)
    for k in range(upper_val,upper_test):
        test_dataset.append(nonfilipino_dataset[k])
        test_label.append(1)

    return train_dataset, train_label, val_dataset, val_label, test_dataset, test_label

def load_author_datasets(format):
    #differentiate comics of 4 authors
    filipino_dataset_path = "dataset/filipino/original/" + format
    abrera_dataset = []
    arre_dataset = []
    borja_dataset = []
    kampilan_dataset = []
    
    #divide dataset to authors
    for file in os.listdir(filipino_dataset_path):
        image = cv2.imread(filipino_dataset_path + "/" + file)
        if file[0:6] == "abrera": abrera_dataset.append(image)
        elif file[0:4] == "arre": arre_dataset.append(image)
        elif file[0:5] == "borja": borja_dataset.append(image)
        elif file[0:8] == "kampilan": kampilan_dataset.append(image)
        else: print("author cannot be identified")

    random.shuffle(abrera_dataset)
    random.shuffle(arre_dataset)
    random.shuffle(borja_dataset)
    random.shuffle(kampilan_dataset)

    #smaller size dataset will be maximum length when dividing
    len_of_dataset = min(len(abrera_dataset),len(arre_dataset),len(borja_dataset),len(kampilan_dataset))

    #set train-validate-test size  at 80:10:10 ratio
    train_size = int(len_of_dataset *0.8) 
    val_size = int(len_of_dataset *0.1)
    test_size = int(len_of_dataset *0.1) 
    print("Ratio of images: train: "+str(train_size)+ " validation: " + str(val_size) + " test_size: "+ str(test_size))
    
    #prepare the training image dataset and its corresponding labels
    train = []
    for k in range(0,train_size):
        train.append([abrera_dataset[k],0])
        train.append([arre_dataset[k],1])
        train.append([borja_dataset[k],2])
        train.append([kampilan_dataset[k],3])
    random.shuffle(train)
    
    train_dataset = []
    train_label = []
    for item, label in train:
        train_dataset.append(item)
        train_label.append(label)


    #prepare the validation image dataset and its corresponding labels
    val_dataset = []
    val_label = []
    upper_val = train_size + val_size
    for k in range(train_size, upper_val): 
        val_dataset.append(abrera_dataset[k])
        val_label.append(0)
    for k in range(train_size, upper_val): 
        val_dataset.append(arre_dataset[k])
        val_label.append(1)
    for k in range(train_size, upper_val): 
        val_dataset.append(borja_dataset[k])
        val_label.append(2)
    for k in range(train_size, upper_val): 
        val_dataset.append(kampilan_dataset[k])
        val_label.append(3)
    
    #prepare the validation image dataset and its corresponding labels
    test_dataset = []
    test_label = []
    upper_test = upper_val + val_size
    for k in range(upper_val,upper_test): 
        test_dataset.append(abrera_dataset[k])
        test_label.append(0)
    for k in range(upper_val,upper_test): 
        test_dataset.append(arre_dataset[k])
        test_label.append(1)
    for k in range(upper_val,upper_test): 
        test_dataset.append(borja_dataset[k])
        test_label.append(2)
    for k in range(upper_val,upper_test): 
        test_dataset.append(kampilan_dataset[k])
        test_label.append(3)


    return train_dataset, train_label, val_dataset, val_label, test_dataset, test_label

def main():
    #use CPU only
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    #LOAD THE DATASET AND DIVIDE TO TRAIN, VALIDATE, AND TEST
    choice = main_menu()
    if choice == "1" : # 2 classes
        format = menu4()
        train, train_label, validate, val_label, test, test_label = load_general_datasets(format)
        units = 2
        activation = "sigmoid"
        loss = "binary_crossentropy"
        categories = ["Filipino", "non-Filipino"]
    elif choice == "2" : # 4 classes
        format = menu4()
        train, train_label, validate, val_label, test, test_label = load_author_datasets(format)
        units = 4
        activation = "softmax"
        loss = "categorical_crossentropy"
        categories = ["abrera", "arre", "borja", "kampilan"]
    else: 
        print("wrong input")
        quit()
    
    #SET RESIZE HERE
    num = 0
    img_size = 200
    for i in range(len(train)):
        train[i] = cv2.resize(train[i],(img_size,img_size))
    for i in range(len(validate)):
        validate[i] = cv2.resize(validate[i],(img_size,img_size))
    for i in range(len(test)):
        test[i] = cv2.resize(test[i],(img_size,img_size))
  

    #PREPARE INPUT DATASET AND LABELS SHAPE AND FORMAT FOR CNN 
    input_size_width = train[0].shape[0]
    input_size_height = train[0].shape[1]
    print(input_size_height)
    train = np.array(train).reshape(-1, input_size_width, input_size_height,3)
    validate = np.array(validate).reshape(-1, input_size_width, input_size_height,3)
    test = np.array(test).reshape(-1, input_size_width, input_size_height,3)
    train_label = to_categorical(train_label)
    val_label = to_categorical(val_label)
    test_label = to_categorical(test_label)

    #MAKE THE CNN MODEL
    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (input_size_width,input_size_height,3)))
    cnn_model.add(MaxPooling2D(pool_size = (2,2), strides = 2 ,padding = "valid"))

    cnn_model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
    cnn_model.add(MaxPooling2D(pool_size = (2,2), strides = 2 ,padding = "valid"))

    cnn_model.add(Conv2D(128, kernel_size = (3,3), activation = "relu"))
    cnn_model.add(MaxPooling2D(pool_size = (2,2), strides = 2 ,padding = "valid"))

    cnn_model.add(Conv2D(256, kernel_size = (3,3), activation = "relu"))
    cnn_model.add(MaxPooling2D(pool_size = (2,2), strides = 2 ,padding = "valid"))

    cnn_model.add(Conv2D(512, kernel_size = (3,3), activation = "relu"))
    cnn_model.add(MaxPooling2D(pool_size = (2,2), strides = 2 ,padding = "valid"))

    cnn_model.add(Flatten())

    cnn_model.add(Dense(units, activation=activation))
    # cnn_model.add(Dense(2, activation="relu"))
    # cnn_model.add(Dense(2, activation="relu"))

    #print CNN model summary
    cnn_model.summary()
    
    # earlystopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 5, restore_best_weights = True)
    #TRAIN DATA
    cnn_model.compile(loss = loss, optimizer = "adam", metrics = ['accuracy'])
    cnn_model.fit(x=train,y=train_label, batch_size = 5, epochs = 10, verbose = 1, validation_data=(validate,val_label )) # validation_data = validate
    # cnn_model.fit(x=train,y=train_label, batch_size = 5, epochs = 25, verbose = 1, validation_data=(validate,val_label ), callbacks = [earlystopping])

    #TEST DATA
    #evaluate model
    test_score = cnn_model.evaluate(test,test_label, verbose = 1) 
    print('Test loss:', test_score[0]) 
    print('Test accuracy:', test_score[1])
    #predict test set
    prediction = cnn_model.predict(test, verbose = 1)

    #SHOW 
    test_label = np.argmax(test_label, axis = 1)
    prediction = np.argmax(prediction, axis = 1)
    #confusion matrix
    conf_matrix = confusion_matrix(test_label, prediction)
    print(conf_matrix)
    show_matrix= ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = categories)
    show_matrix.plot()
    plt.show()
    #general report
    print(classification_report(test_label, prediction))

main()