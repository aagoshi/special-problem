'''
IMAGE PRE-PROCESSING FOR COMICS CLASSIFICATION
'''
import cv2
import numpy as np
import os
import shutil
import random
from matplotlib import pyplot as plt

#menu for author name
def menu2():
    print(
    '''
    ---------------------MENU----------------------------
    [1] Abrera
    [2] Arre
    [3] Borja
    [4] Kampilan
    [5] EBDtheque 
    -----------------------------------------------------  
    ''')
    choice = input("choose from menu: ")
    if choice == "1" : choice = "abrera"
    elif choice == "2" : choice = "arre"
    elif choice == "3" : choice = "borja"
    elif choice == "4" : choice = "kampilan"
    elif choice == "5": choice = "ebdtheque"
    else: print("wrong input...")
    return choice

#menu for filipino/ nonfilipino dataset folder
def menu3():
    print(
    '''
    ---------------------MENU----------------------------
    [1] Filipino dataset
    [2] Non-Filipino dataset
    -----------------------------------------------------  
    ''')
    choice = input("choose from menu: ")
    if choice == "1" : choice = "filipino"
    elif choice == "2" : choice = "nonfilipino"
    else: print("wrong input...")
    return choice

#menu for diff preprocess folder format
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

#main menu
def main_menu():
    print(
    '''
    ---------------------MENU----------------------------
    [1] Segment panels and move to raw-initial
    [2] Move to working dataset folder 
    [3] Get lineart
    [4] Get Histogram
    -----------------------------------------------------  
    ''')
    choice = input("choose from menu: ")    
    return choice

#------main programs----------#
def view_image(img):
    cv2.imshow("test",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_connected_components(image, author, file):
    #pre-processing for connected components
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    filt_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    # threshold= cv2.threshold(filt_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    threshold= cv2.threshold(filt_img, 200, 255, cv2.THRESH_BINARY_INV)[1] #borja only special case since bounding box of a panel is light

    # use connected components to identify lines connected
    cc = cv2.connectedComponentsWithStats(threshold,4, cv2.CV_32S)
    (numOfComponents, label_ids, values, centroids) = cc

    output = np.zeros(gray_img.shape, dtype="uint8")

    num = 0
    #identify which component is a panel
    for i in range(1, numOfComponents):
   
        # Area of the component
        area = values[i, cv2.CC_STAT_AREA]
        
        if (area > 100000) and (area < 10000000000000):
            # Create a new image for bounding boxes
            new_img=image.copy()
            
            # Now extract the coordinate points
            x1 = values[i, cv2.CC_STAT_LEFT]
            y1 = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]
            
            # Coordinate of the bounding box
            x2 = x1 + w
            y2 = y1 + h
            (X, Y) = centroids[i]

            #slice image to panel location only
            panel = new_img[y1:y2, x1:x2]

            # view_image(panel)
            num+=1
            write_path = "dataset/raw-initial/" + author + "/so/" + file[:-4]+ "_"+ str(num) + ".jpg"
            cv2.imwrite(write_path, panel)


    return 0

# #gets a comic page and returns an array of panels in that page
def get_single_panel(image, height, width, author, file):
    get_connected_components(image,author, file)
    return 0

#edge mask gets a binary image for lineart
def line_art(image,height, width, threshold):

    #create blank binary image
    bin_img = np.zeros((height,width), dtype = np.uint8)
    bin_img.fill(255)
    #prepare for binarization
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    filt_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    # edge_img = cv2.Canny(filt_img, 20, 200)
    # eb = 8
    # is_edge = False

    #BINARIZATION
    for x in range(0, height-1):
        for y in range(0, width-1):
            if(filt_img[x][y] < threshold): # if pixel is dark accdng to threshold, it will be colored black else remain white
                bin_img[x][y] = 0
                # intended to use this buffer so that edge can be considered, but found that simple binarization provides more natural output
                # for k in range(x-eb,x+eb): 
                #     if ((k< height-1) and (edge_img[k][y] == 255)): 
                #         is_edge = True 
                #         break
                # if(is_edge): 
                #     bin_img[x][y] = 0
                #     is_edge = False
    
    # view_image(bin_img)

    return bin_img

#ayusin histogram
#generate an RGB histogram from image
def RGB_histogram(image, height, width):
    #create blank hist image
    hist_height = int((height * width)/10)
    hist_img = np.zeros((hist_height,255,3), dtype = np.uint8)
    hist_img.fill(0)

    red_histogram_tracker = [0 for k in range(256)] #tracks height of histogram
    green_histogram_tracker = [0 for k in range(256)] #tracks height of histogram
    blue_histogram_tracker = [0 for k in range(256)] #tracks height of histogram

    #counts the red, green, and blue values
    for x in range(0,height):
        for y in range(0,width):
            red = image[x,y][2]
            green = image[x,y][1]
            blue = image[x,y][0]
            red_histogram_tracker[red] +=1
            green_histogram_tracker[green] +=1
            blue_histogram_tracker[blue] +=1
    
    #histogram for red channel
    for x in range(0,255):
        for y in range(0,red_histogram_tracker[x]):
            if (y < hist_height):
                hist_img[y][x] = [0,0,255]

    #histogram for green channel
    for x in range(0,255):
        for y in range(0,green_histogram_tracker[x]):
            if (y < hist_height):
                hist_img[y][x] = [0,255,0]

    #histogram for blue channel
    for x in range(0,255):
        for y in range(0,blue_histogram_tracker[x]):
            if (y < hist_height):
                hist_img[y][x] = [255,0,0]


    #scale down and rotate histogram
    resize_hist = cv2.resize(hist_img,(255,255))   
    flip_hist = cv2.flip(resize_hist,0)
    
    
    return flip_hist


def initial_preprocess(author):
    dataset_path = "dataset/raw/" + author
    l= 0
    for file in os.listdir(dataset_path):
        file_path = dataset_path + "/" + file
        image = cv2.imread(file_path)
        imheight = image.shape[0]
        imwidth = image.shape[1]

        #write wo/ whole page-original
        write_path = "dataset/raw-initial/" + author + "/wo/"+ file
        cv2.imwrite(write_path, image)

        #get single panel
        get_single_panel(image,imheight, imwidth, author, file) 

    return 0

#move to working dataset
def move_working_dataset(author):
    if author == "ebdtheque": author_class = "nonfilipino" 
    else: author_class = "filipino"

    #move whole page
    dataset_path = "dataset/raw-initial/" + author +"/wo"
    num = 0
    for file in os.listdir(dataset_path):
        num+=1
        file_path = dataset_path + "/" + file
        dest_path = "dataset/" + author_class + "/original/wo/" + author + "_" + str(num) + ".jpg"
        shutil.move(file_path, dest_path)
    
    #move single panels
    dataset_path = "dataset/raw-initial/" + author +"/so"
    num = 0
    for file in os.listdir(dataset_path):
        num+=1
        file_path = dataset_path + "/" + file
        dest_path = "dataset/" + author_class + "/original/so/" + author + "_" + str(num) + ".jpg"
        shutil.move(file_path, dest_path)
    return 0

def get_line_art(author_class):
    dataset_path = "dataset/" + author_class + "/original/wo"
    spec_ebdtheque = ["ebdtheque_18.jpg","ebdtheque_19.jpg", "ebdtheque_20.jpg", "ebdtheque_57.jpg", "ebdtheque_58.jpg", "ebdtheque_59.jpg", "ebdtheque_60.jpg", "ebdtheque_61.jpg", "ebdtheque_62.jpg", "ebdtheque_63.jpg", "ebdtheque_64.jpg", "ebdtheque_65.jpg"] 
    for file in os.listdir(dataset_path):
        file_path = dataset_path + "/" + file
        image = cv2.imread(file_path)
        imheight = image.shape[0]
        imwidth = image.shape[1]
        if file[0:3] == "abr": threshold = 50
        elif file[0:3] == "arr": threshold = 200
        elif file in spec_ebdtheque: threshold = 210
        else: threshold = 100
        line_img = line_art(image,imheight, imwidth, threshold)
        write_path = "dataset/"+ author_class + "/original/wl/" + file
        cv2.imwrite(write_path, line_img)
    
    #for single panel
    dataset_path = "dataset/" + author_class + "/original/so" 
    for file in os.listdir(dataset_path):
        file_path = dataset_path + "/" + file
        image = cv2.imread(file_path)
        imheight = image.shape[0]
        imwidth = image.shape[1]
        if file[0:3] == "abr": threshold = 50
        elif file[0:3] == "arr": threshold = 200
        elif file in spec_ebdtheque: threshold = 210
        else: threshold = 100
        line_img = line_art(image,imheight, imwidth, threshold)
        write_path = "dataset/"+ author_class + "/original/sl/" + file
        cv2.imwrite(write_path, line_img)

    return 0

def get_histogram(author_class):
    #for whole page
    dataset_path = "dataset/" + author_class + "/original/wo"
    for file in os.listdir(dataset_path):
        file_path = dataset_path + "/" + file
        image = cv2.imread(file_path)
        imheight = image.shape[0]
        imwidth = image.shape[1]
        hist_img = RGB_histogram(image,imheight, imwidth)
        write_path = "dataset/"+ author_class + "/original/wh/" + file
        cv2.imwrite(write_path, hist_img)
    
    #for single panel
    dataset_path = "dataset/" + author_class + "/original/so" 
    for file in os.listdir(dataset_path):
        file_path = dataset_path + "/" + file
        image = cv2.imread(file_path)
        imheight = image.shape[0]
        imwidth = image.shape[1]
        hist_img = RGB_histogram(image,imheight, imwidth)
        write_path = "dataset/"+ author_class + "/original/sh/" + file
        cv2.imwrite(write_path, hist_img)
    
    return 0

def shuffle_dataset(author_class,format):
    dataset_path = "dataset/" + author_class + "/original/" + format
    list_files = os.listdir(dataset_path)
    random.shuffle(list_files)

    if format[0] == "w": 
        train_dataset = list_files[0:56] #56
        val_dataset = list_files[56:63] #7
        test_dataset = list_files[63:70] #7
    elif format[0] == "s": #use only 220 images
        train_dataset = list_files[0:100] #176
        val_dataset = list_files[100:120] #22
        test_dataset = list_files[120:140] #22
    else: print("wrong format...")

    for file in train_dataset:
        file_path = dataset_path + "/" + file
        dest_path = "dataset/" + author_class + "/train/"+ format + "/" +file
        shutil.move(file_path, dest_path)

    for file in val_dataset:
        file_path = dataset_path + "/" + file
        dest_path = "dataset/" + author_class + "/validate/"+ format + "/" + file
        shutil.move(file_path, dest_path)

    for file in test_dataset:
        file_path = dataset_path + "/" + file
        dest_path = "dataset/" + author_class + "/test/"+ format + "/" + file
        shutil.move(file_path, dest_path)

    return 0

def main():
    choice = main_menu()
    if (choice == "1" ): 
        author = menu2()
        initial_preprocess(author)
    elif (choice == "2"): 
        author = menu2()
        move_working_dataset(author)
    elif (choice == "3"): 
        author_class = menu3()
        get_line_art(author_class)
    elif (choice == "4"):
        author_class = menu3()
        get_histogram(author_class)
    elif (choice == "5"):
        author_class = menu3()
        format= menu4()
        shuffle_dataset(author_class,format)
    else: print("wrong input")
    return 0

main()
