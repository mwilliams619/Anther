from django.shortcuts import redirect
from django.urls import reverse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login

class LoginMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.user.is_authenticated:
            # If the user is not authenticated (logged in), redirect them to the login page
            login_url = ['/login/register/', '/login/']  # Remove '/login/' from the list
            if request.path not in login_url and not request.path.startswith('/admin/'):
                return redirect(login_url[1])

        response = self.get_response(request)
        print(response)

        # Code to be executed for each request/response after the view is called
        return response
