from django.db import models
from django import forms

# Create your models here.

DETAIL_CHOICES = [('high', 'High Level'),
                  ('low', 'Low Level'),
                  ('both', 'Both'),
                  ]

class processText(forms.Form):
    detail_level = forms.CharField(label='Desired detail', widget=forms.RadioSelect(choices=DETAIL_CHOICES))

    input_text   = forms.CharField(label='', widget=forms.Textarea(attrs={'placeholder':'input_text'}))
