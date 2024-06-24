import os
import cv2
import numpy as np
from sklearn import preprocessing
from progress.bar import Bar
import time

def main():
    mainStartTime = time.time()
    trainImagePath = './images_split/train/'
    testImagePath = './images_split/test/'
    trainFeaturePath = './features_labels/gabor/train/'
    testFeaturePath = './features_labels/gabor/test/'

    # Criar diretórios se não existirem
    os.makedirs(trainFeaturePath, exist_ok=True)
    os.makedirs(testFeaturePath, exist_ok=True)

    print(f'[INFO] ========= TRAINING IMAGES ========= ')
    trainImages, trainLabels = getData(trainImagePath)
    trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
    trainFeatures = extractGaborFeatures(trainImages)
    saveData(trainFeaturePath,trainEncodedLabels,trainFeatures,encoderClasses)
    print(f'[INFO] =========== TEST IMAGES =========== ')
    testImages, testLabels = getData(testImagePath)
    testEncodedLabels, encoderClasses = encodeLabels(testLabels)
    testFeatures = extractGaborFeatures(testImages)
    saveData(testFeaturePath,testEncodedLabels,testFeatures,encoderClasses)
    elapsedTime = round(time.time() - mainStartTime,2)
    print(f'[INFO] Code execution time: {elapsedTime}s')

def getData(path):
    images = []
    labels = []
    if os.path.exists(path):
        for dirpath , dirnames , filenames in os.walk(path):   
            if (len(filenames)>0): #it's inside a folder with files
                folder_name = os.path.basename(dirpath)
                bar = Bar(f'[INFO] Getting images and labels from {folder_name}',max=len(filenames),suffix='%(index)d/%(max)d Duration:%(elapsed)ds')            
                for index, file in enumerate(filenames):
                    label = folder_name
                    labels.append(label)
                    full_path = os.path.join(dirpath,file)
                    image = cv2.imread(full_path)
                    images.append(image)
                    bar.next()
                bar.finish()
        return images, np.array(labels,dtype=object)
    
def extractGaborFeatures(images):
    bar = Bar('[INFO] Extracting Gabor features...', max=len(images), suffix='%(index)d/%(max)d Duration:%(elapsed)ds')
    featuresList = []
    for image in images:
        if len(image.shape) > 2:  # imagem colorida
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filters = buildGaborFilters()
        features = applyGaborFilters(image, filters)
        featuresList.append(features)
        bar.next()
    bar.finish()
    return np.array(featuresList, dtype=object)

def buildGaborFilters():
    filters = []
    ksize = 31  # tamanho do kernel
    for theta in np.arange(0, np.pi, np.pi / 4):  # 4 orientações
        for sigma in (1, 3):  # 2 valores de sigma
            for lamda in np.arange(np.pi / 4, np.pi, np.pi / 4):  # 4 frequências
                kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
                filters.append(kern)
    return filters

def applyGaborFilters(img, filters):
    responses = []
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        responses.append(fimg.mean())
    return np.array(responses)

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
    # Criar diretório se não existir
    os.makedirs(path, exist_ok=True)
    label_filename = f'{labels=}'.split('=')[0]+'.csv'
    feature_filename = f'{features=}'.split('=')[0]+'.csv'
    encoder_filename = f'{encoderClasses=}'.split('=')[0]+'.csv'
    np.savetxt(path+label_filename, labels, delimiter=',', fmt='%i')
    np.savetxt(path+feature_filename, features, delimiter=',')  # float does not need format
    np.savetxt(path+encoder_filename, encoderClasses, delimiter=',', fmt='%s')
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Saving done in {elapsedTime}s')

if __name__ == "__main__":
    main()
