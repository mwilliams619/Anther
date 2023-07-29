from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.views.decorators.csrf import csrf_exempt
from Anther.models import MailingList
import re

def login_view(request):
    error_message = ""  # Initialize the error_message variable
    if request.method == 'POST':
        # Get the username/email and password from the form
        username_or_email = request.POST.get('username_or_email')
        password = request.POST.get('password')

        # Authenticate the user
        user = authenticate(request, username=username_or_email, password=password)

        # Check if the authentication was successful
        if user is not None:
            # Log in the user
            login(request, user)
            # Redirect to a success page (replace 'home' with the name of your home page URL pattern)
            return redirect('home')
        else:
            # Authentication failed, show an error message or handle it as per your requirement
            error_message = "Invalid credentials. Please try again."

    # If the request method is GET or authentication failed, render the login template
    return render(request, 'login.html', {'error_message': error_message})

def registration_view(request):
    error_message = ""  # Initialize the error_message variable
    if request.method == 'POST':
        
        # Get the registration form data
        first_name = request.POST.get('first-name')
        last_name = request.POST.get('last-name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        subscribe_newsletter = request.POST.get('subscribed', False)  # Set default to False if not provided

        # Check if a user with the provided email already exists
        if is_valid_email(email):

            if User.objects.filter(email=email).exists():
                error_message = "A user with this email already exists."
                return render(request, 'login.html', {'error_message': error_message})
            else:
                # Create a new user object
                user = User.objects.create_user(
                    username=email,  # Use email as the username
                    email=email,
                    password=password,
                    first_name=first_name,
                    last_name=last_name
                )
        else:
            error_message = "Invalid email address"
            return render(request, 'login.html', {'error_message': error_message})

        # Optionally, you can log in the user after registration
        login(request, user)

        # Save the user object to the database
        user.save()

        # Redirect to a success page (replace 'home' with the name of your home page URL pattern)
        return redirect('home')

    # If the request method is GET, render the registration template
    return render(request, 'login.html', {'error_message': None})

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email)

@csrf_exempt
def add_to_mailing_list(request):
    if request.method == 'POST':
        email = request.POST.get('email')

        if is_valid_email(email):
            # Check if the email already exists in the MailingList table
            if MailingList.objects.filter(email=email).exists():
                return JsonResponse({'message': 'Email already subscribed.'}, status=400)

            # If the email is not already in the table, add it
            mailing_list_entry = MailingList(email=email)
            mailing_list_entry.save()

            return JsonResponse({'message': 'Email subscribed successfully.'}, status=200)
        else:
            return JsonResponse({'message': 'Invalid email address.'}, status=400)

    return JsonResponse({'message': 'Invalid request method.'}, status=405)
