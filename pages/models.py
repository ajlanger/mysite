from django.db import models
from django import forms

# Create your models here.
class ContactForm(forms.Form):
    name = forms.CharField(max_length=500)
