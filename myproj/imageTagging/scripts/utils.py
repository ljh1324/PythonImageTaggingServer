def surePredict(predictionList):
  shape = predictionList.shape
  surePredictionList = []

  for i in range(shape[0]):
    maxValue = predictionList[i][0]
    maxIdx = 0
    for j in range(1, shape[1]):
      if maxValue < predictionList[i][j]:
        maxValue = predictionList[i][j]
        maxIdx = j
    
    print(maxValue)
    if (maxValue >= 0.7):
      surePredictionList.append(maxIdx)
  
  return surePredictionList
    