import numpy as np


def seam_carve (img, mode, mask) :
    yuvRGB = np.array([0.299, 0.587, 0.114])
    newIMG = img.dot(yuvRGB)

    gradX = np.zeros((img.shape[0], img.shape[1]))
    for i in range(gradX.shape[0]):
        gradX[i] = (newIMG[min(i + 1, gradX.shape[0] - 1)] - newIMG[max(0, i - 1)])
    
    gradY = np.zeros((img.shape[1], img.shape[0]))
    newIMG = np.transpose(newIMG)

    for i in range(gradY.shape[0]):
        gradY[i] = (newIMG[min(i + 1, gradY.shape[0] - 1)] - newIMG[max(0, i - 1)])
    
    gradIMG = np.sqrt(gradX ** 2 + np.transpose(gradY) ** 2)

    if  mask is not None:
        gradIMG += (mask * (mask.shape[0] * mask.shape[1] * 256))
    
    mode = list(mode.split(' '))
    if mode[0][0] == 'h':
        if mode[1][0] == 's':
            return shrinking(img, mask, gradIMG)
        else:
            return expanding(img, mask, gradIMG)
    else:
        if mask is None:
            r = [np.transpose(img, (1, 0, 2)), None, np.transpose(gradIMG)]
        else:
            r = [np.transpose(img, (1, 0, 2)), np.transpose(mask), np.transpose(gradIMG)]
        
        if mode[1][0] == 's':
            r = shrinking(r[0], r[1], r[2])
        else:
            r = expanding(r[0], r[1], r[2])

        if mask is None:
            return np.transpose(r[0], (1, 0, 2)), None, np.transpose(r[2])
        else:
            return np.transpose(r[0], (1, 0, 2)), np.transpose(r[1]), np.transpose(r[2])


def shrinking(img, mask, gradIMG):
    newMask = np.zeros((img.shape[0], img.shape[1]))
    newIMG = np.zeros((img.shape[0], img.shape[1] - 1, 3))
    if mask is not None:
        changedMask = np.zeros((img.shape[0], img.shape[1] - 1))

    for i in range(1, gradIMG.shape[0]):
        for j in range(gradIMG.shape[1]):
            gradIMG[i][j] += min(gradIMG[i - 1, max(0, j - 1) : min(j + 2, gradIMG.shape[1])])
    
    idx = gradIMG[-1].argmin()

    for i in range(gradIMG.shape[0] - 1, -1, -1):
        newMask[i][idx] = 1
        newIMG[i] = np.delete(img[i], idx, axis=0)
        if not (mask is None):
            changedMask[i] = np.delete(mask[i], idx, axis=0)

        step = gradIMG[i - 1, max(idx - 1, 0) : idx + 2]
        if idx != 0:
            idx -= 1
        idx += step.argmin()

    if mask is None:
        return newIMG, None, newMask
    else:
        return newIMG, changedMask, newMask


def expanding(img, mask, gradIMG):
    newIMG = np.zeros((img.shape[0], img.shape[1] + 1, 3))
    newMask = np.zeros((img.shape[0], img.shape[1]))
    if mask is not None:
        changedMask = np.zeros((img.shape[0], img.shape[1] + 1))

    for i in range(1, gradIMG.shape[0]):
        for j in range(gradIMG.shape[1]):
            gradIMG[i][j] += min(gradIMG[i - 1, max(0, j - 1) : min(j + 2, gradIMG.shape[1])])
    
    idx = gradIMG[-1].argmin()

    for i in range(gradIMG.shape[0] - 1, -1, -1):
        newMask[i][idx] = 1
        newIMG[i] = np.insert(img[i], idx + 1, (img[i][idx] + img[i][min(idx + 1, img.shape[1] - 1)]) / 2, axis=0)
        if not (mask is None):
            changedMask[i] = np.insert(mask[i], idx, (mask[i][idx] + mask[i][min(idx + 1, img.shape[1] - 1)]) / 2, axis=0)

        step = gradIMG[i - 1, max(idx - 1, 0) : idx + 2]
        if idx != 0:
            idx -= 1
        idx += step.argmin()

    if mask is None:
        return newIMG, None, newMask
    else:
        return newIMG, changedMask, newMask
