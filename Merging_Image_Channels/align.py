import numpy as np
from skimage.transform import rescale


def MSE(i1, i2):
    return ((i1 - i2) ** 2).sum() / (i1.shape[0] * i1.shape[1])


def cross_correlation(i1, i2):
    return (i1 * i2).sum() / (((i1 ** 2).sum() * (i2 ** 2).sum()) ** 0.5)


def lowMetric(newImg, baseImg, offset=15, func=MSE):
    allFuncs = [MSE, cross_correlation]
    if func not in allFuncs:
        print("Wrong function in lowMetric")
        return
    
    lookingFunction = min
    bestMetric = np.inf
    if func is cross_correlation:
        lookingFunction = max
        bestMetric = -1
    
    bestHeight = -offset
    bestWidth = -offset
    
    for shiftH in range(-offset, offset + 1):
        for shiftW in range(-offset, offset + 1):
            shiftedBaseImg = baseImg[max(shiftH, 0) : baseImg.shape[0] + shiftH, max(shiftW, 0) : baseImg.shape[1] + shiftW]
            shiftedNewImg = newImg[max(-shiftH, 0) : newImg.shape[0] - shiftH, max(-shiftW, 0) : newImg.shape[1] - shiftW]
            newMetric = func(shiftedBaseImg, shiftedNewImg)
            if bestMetric != lookingFunction(newMetric, bestMetric):
                bestMetric = lookingFunction(newMetric, bestMetric)
                bestHeight = shiftH
                bestWidth = shiftW
    return (bestHeight, bestWidth)


def highMetric(newImg, baseImg, func=MSE):
    k = 2
    h = newImg.shape[0]
    w = newImg.shape[1]
    newImgMod = rescale(newImg, 0.5)
    baseImgMod = rescale(baseImg, 0.5)
    while (h / k) > 500 or (w / k) > 500:
        newImgMod = rescale(newImgMod, 0.5)
        baseImgMod = rescale(baseImgMod, 0.5)
        k *= 2
    bestHeight, bestWidth = lowMetric(newImgMod, baseImgMod)

    shiftedBaseImg = baseImg[max(bestHeight * k, 0) : baseImg.shape[0] + bestHeight * k, max(bestWidth * k, 0) : baseImg.shape[1] + bestWidth * k]
    shiftedNewImg = newImg[max(-bestHeight * k, 0) : newImg.shape[0] - bestHeight * k, max(-bestWidth * k, 0) : newImg.shape[1] - bestWidth * k]

    bestHeightZoomed, bestWidthZoomed = lowMetric(shiftedNewImg, shiftedBaseImg, offset=3)
    return (bestHeight * k + bestHeightZoomed, bestWidth * k + bestWidthZoomed)


def align(img, green_coord):
    """
    Blue
    Green
    Red
    """
    imgHeight, imgWidth = img.shape[0], img.shape[1]
    
    height = imgHeight // 3
    width = imgWidth

    frameHeight = int(height * 0.05)
    frameWidth = int(width * 0.05)

    blueImg = img[frameHeight : height - frameHeight, frameWidth : width - frameWidth]
    greenImg = img[frameHeight + height : height * 2 - frameHeight, frameWidth : width - frameWidth]
    redImg = img[frameHeight + height * 2 : height * 3 - frameHeight, frameWidth : width - frameWidth]

    metricFunc = highMetric
    if blueImg.shape[0] < 500 and blueImg.shape[1] < 500:
        metricFunc = lowMetric

    blueShift = metricFunc(blueImg, greenImg)
    redShift = metricFunc(redImg, greenImg)

    blue_coord = (green_coord[0] - height - blueShift[0], green_coord[1] - blueShift[1])
    red_coord = (green_coord[0] + height - redShift[0], green_coord[1] - redShift[1])

    red_image = np.roll(redImg, redShift[0], axis = 0)
    red_image = np.roll(red_image, redShift[1], axis = 1)
    blue_image = np.roll(blueImg, blueShift[0], axis = 0)
    blue_image = np.roll(blue_image, blueShift[1], axis = 1)

    outputImg = np.dstack([red_image, greenImg, blue_image])

    return outputImg, blue_coord, red_coord
