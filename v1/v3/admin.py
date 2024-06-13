from django.contrib import admin
from .models import UserModel,Video,UserRank

admin.site.register(UserModel)
admin.site.register(Video)
admin.site.register(UserRank)