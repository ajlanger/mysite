from django.shortcuts import render

# Create your views here.
def home_view(request, *args, **kwargs): # *args, **kwargs
    return render(request, 'home.html', {}) # home.html is a template

def cv_view(request, *args, **kwargs):
    return render(request, 'CV-html.html', {})
