from django.shortcuts import render, redirect
from .models import ContactForm
from django.core.mail import send_mail
from django.http import HttpResponse, HttpResponseRedirect

# Create your views here.
def home_view(request, *args, **kwargs): # *args, **kwargs
    name    = ""
    email   = ""
    comment = ""

    form    = ContactForm(request.POST or None)
    if form.is_valid():
        name    = form.cleaned_data.get("name")
        email   = form.cleaned_data.get("email")
        comment = form.cleaned_data.get("comment")

        if request.user.is_authenticated:
            subject = str(request.user) + "'s Comment/question"
        else:
            subject = f"{name}'s Comment/question"

        # Send mail to me
        message_to_me = f"Hello sir, you received a new message.\n\n Name: {name} \nEmail: {email} \nComment/Question: {comment} \n\nHave a nice day!"

        send_mail(subject, message_to_me, 'smtp.gmail.com', ['langeraertarnaud@gmail.com'])

        # Send confirmation mail to sender
        subject = 'Message to Arnaud'
        message_to_user = f"Dear {name}, \n \nYour message was sent with success. Arnaud will try to answer as soon as he can. \n \nThis was your message: \n {comment}"
        send_mail(subject, message_to_user, 'smtp.gmail.com', [email])

        return HttpResponseRedirect('/')
    else:
        context = {'form': form}
        return render(request, 'home.html', context)

def cv_view(request, *args, **kwargs):
    return render(request, 'CV-html.html', {})
