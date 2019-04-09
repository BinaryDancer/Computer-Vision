import math
import numpy as np
from skimage import transform
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle


def fit_and_classify(X_train, y_train, X_test):
    X, y = shuffle(X_train, y_train, random_state=0)
    svm_clf = LinearSVC(C=3.65)
    svm_clf.fit(X, y)
    return svm_clf.predict(X_test)


def extract_hog(img):
    imgSize = 80
    gradIMG, gradDirection = getGrad(img, imgSize)
    
    cellSize = 8
    binCount = 8
    blocksAndSteps = [[4, 2], [7, 3]]

    HOGmatrix = np.zeros((imgSize // cellSize, imgSize // cellSize, binCount))
    for i in range(imgSize):
        for j in range(imgSize):
            binN = int((gradDirection[i][j] * binCount) / math.pi) % binCount
            HOGmatrix[i // cellSize][j // cellSize][binN] += gradIMG[i][j]

    vectorHOG = np.array([])
    for blockSize, stepSize in blocksAndSteps:
        for i in range(0, HOGmatrix.shape[0] - blockSize + 1, stepSize):
            for j in range(0, HOGmatrix.shape[1] - blockSize + 1, stepSize):
                vectorHOG = np.append(vectorHOG, normalize(np.ravel(HOGmatrix[i : i + blockSize][j : j + blockSize])))

    return vectorHOG.ravel()


def normalize(vector):
    return vector / math.sqrt((vector ** 2).sum() + 1e-10)


def getGrad(img, imgSize):
    yuvRGB = np.array([0.299, 0.587, 0.114])
    newIMG = img.dot(yuvRGB)
    newIMG = transform.resize(newIMG, [imgSize, imgSize])

    gradX = np.zeros((newIMG.shape[0], newIMG.shape[1]))
    for i in range(gradX.shape[0]):
        gradX[i] = (newIMG[min(i + 1, gradX.shape[0] - 1)] - newIMG[max(0, i - 1)])
    
    gradY = np.zeros((newIMG.shape[1], newIMG.shape[0]))
    newIMG = np.transpose(newIMG)

    for i in range(gradY.shape[0]):
        gradY[i] = (newIMG[min(i + 1, gradY.shape[0] - 1)] - newIMG[max(0, i - 1)])
    
    gradY = np.transpose(gradY)

    gradIMG = np.sqrt(gradX ** 2 + gradY ** 2)
    directionIMG = np.abs(np.arctan2(gradY, gradX))
    return gradIMG, directionIMG