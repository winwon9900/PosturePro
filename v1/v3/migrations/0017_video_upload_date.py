# Generated by Django 5.0.6 on 2024-06-09 11:10

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('v3', '0016_video_alter_document_title'),
    ]

    operations = [
        migrations.AddField(
            model_name='video',
            name='upload_date',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
    ]
