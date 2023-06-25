from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from poker_royale.models import Machine
from .models import Profile

class UserRegisterForm(UserCreationForm):
	email = forms.EmailField(required=True)

	class Meta:
		model = User
		fields = ['username', 'email', 'password1', 'password2'] 

class UserUpdateForm(forms.ModelForm):
	email = forms.EmailField()

	class Meta:
		model = User
		fields = ['username', 'email']

class ProfileUpdateForm(forms.ModelForm):
	class Meta:
		model = Profile
		fields = ['machine'] 
	def __init__(self, *args, **kwargs):
		machine_set = kwargs.pop('machine_set', None)
		super(ProfileUpdateForm, self).__init__(*args, **kwargs)
		self.fields['machine'].queryset = machine_set