from django.shortcuts import redirect
from django.shortcuts import render, get_object_or_404
from django.utils import timezone
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

import json
import numpy as np
import tensorflow as tf

global graph, model

from PIL import Image

from .scripts import models
from .scripts import defines

graph = tf.get_default_graph()
model = models.model(defines.IMAGE_SIZE, defines.LABEL_SIZE)

# Create your views here.
def index(request):
  return HttpResponse('hello')

@csrf_exempt
def image(request):
  if request.method == "POST":
    print(request.FILES)
    print(request.FILES["files[]"])
    print(request.POST)

    dataList = []
    for f in request.FILES.getlist('files[]'):
      img = Image.open(f)
      resizedImg = img.resize((256, 256), Image.ANTIALIAS)
      imgNumpy = np.asarray(resizedImg)
      dataList.append(imgNumpy)
      print(img)
      print(imgNumpy)

    dataList = np.array(dataList)
    
    with graph.as_default():
      predicted = model.predict(x=dataList)
      predicted = np.argmax(predicted, 1)
      print(predicted)

    return HttpResponse('image class')
  else:
    return HttpResponse('nope')