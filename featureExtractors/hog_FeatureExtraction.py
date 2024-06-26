import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn import preprocessing
from progress.bar import Bar
import time

def main():
    mainStartTime = time.time()
    trainImagePath = './images_split/train/'
    testImagePath = './images_split/test/'
    trainFeaturePath = './features_labels/hog/train/'
    testFeaturePath = './features_labels/hog/test/'
    print(f'[INFO] ========= TRAINING IMAGES ========= ')
    trainImages, trainLabels = getData(trainImagePath)
    trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
    trainFeatures = extractHOGFeatures(trainImages, size=(256, 256))  # Adjust size here
    saveData(trainFeaturePath, trainEncodedLabels, trainFeatures, encoderClasses)
    print(f'[INFO] =========== TEST IMAGES =========== ')
    testImages, testLabels = getData(testImagePath)
    testEncodedLabels, encoderClasses = encodeLabels(testLabels)
    testFeatures = extractHOGFeatures(testImages, size=(256, 256))  # Adjust size here
    saveData(testFeaturePath, testEncodedLabels, testFeatures, encoderClasses)
    elapsedTime = round(time.time() - mainStartTime, 2)
    print(f'[INFO] Code execution time: {elapsedTime}s')

def getData(path):
    images = []
    labels = []
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):   
            if len(filenames) > 0:  # it's inside a folder with files
                folder_name = os.path.basename(dirpath)
                bar = Bar(f'[INFO] Getting images and labels from {folder_name}', max=len(filenames), suffix='%(index)d/%(max)d Duration:%(elapsed)ds')            
                for index, file in enumerate(filenames):
                    label = folder_name
                    labels.append(label)
                    full_path = os.path.join(dirpath, file)
                    image = cv2.imread(full_path)
                    images.append(image)
                    bar.next()
                bar.finish()
        return images, np.array(labels, dtype=object)

def extractHOGFeatures(images, size=(256, 256)):
    bar = Bar('[INFO] Extracting HOG features...', max=len(images), suffix='%(index)d/%(max)d  Duration:%(elapsed)ds')
    featuresList = []
    for image in images:
        if len(image.shape) > 2:  # Color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, size)  # Resize image to a consistent size
        features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        featuresList.append(features)
        bar.next()
    bar.finish()
    return np.array(featuresList, dtype=object)

def encodeLabels(labels):
    startTime = time.time()
    print(f'[INFO] Encoding labels to numerical labels')
    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Encoding done in {elapsedTime}s')
    return np.array(encoded_labels, dtype=object), encoder.classes_

def saveData(path, labels, features, encoderClasses):
    startTime = time.time()
    print(f'[INFO] Saving data')
    label_filename = 'labels.csv'
    feature_filename = 'features.csv'
    encoder_filename = 'encoderClasses.csv'
    os.makedirs(path, exist_ok=True)
    np.savetxt(os.path.join(path, label_filename), labels, delimiter=',', fmt='%i')
    np.savetxt(os.path.join(path, feature_filename), features, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(path, encoder_filename), encoderClasses, delimiter=',', fmt='%s')
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Saving done in {elapsedTime}s')

if __name__ == "__main__":
    main()
