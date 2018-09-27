from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),  # post_list라는 이름의 view가 ^$ URL에 할당됨.
    url(r'^image$', views.image, name='image')
]
