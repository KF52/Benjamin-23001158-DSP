# Generated by Django 5.1.6 on 2025-04-22 21:52

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0015_prediction_confidence_score_prediction_loan_approval'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='prediction',
            name='credit_score',
        ),
    ]
