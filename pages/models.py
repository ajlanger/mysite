from django.db import models
from django import forms
from phone_field import PhoneField

# Create your models here.
class ContactForm(forms.Form):
    name    = forms.CharField(max_length=500, label='', widget=forms.TextInput(attrs={'placeholder':'Name'}))
    email   = forms.EmailField(max_length=500, label='', widget=forms.TextInput(attrs={'placeholder':'Email'}))
    comment = forms.CharField(label='', widget=forms.Textarea(attrs={'placeholder':'Message'}))
