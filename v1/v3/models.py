from django.db import models

class UserModel(models.Model):
    class Meta: # 데이터베이스 모델 정보가 담긴 공간
        db_table="my-user"
    username = models.CharField(max_length=20, null=False, primary_key=True)
    password = models.CharField(max_length=100, null=False)
    password2 = models.CharField(max_length=100, null=False,default="")

class UserRank(models.Model):
    class Meta: # 데이터베이스 모델 정보가 담긴 공간
        db_table="my-user_rank"
    rank = models.CharField(max_length=100, default='')

    def get_rank_value(self):
        return int(self.rank)
class Video(models.Model):
    title = models.CharField(max_length=200)
    document = models.FileField(upload_to='videos/')
    upload_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title