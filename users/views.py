from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UserRegisterForm, UserUpdateForm, ProfileUpdateForm
from kdata_tf.kdata_tf_lib import delete_machine

def register(request):
	if request.method == "POST":
		form = UserRegisterForm(request.POST)
		if form.is_valid():
			form.save()
			username = form.cleaned_data.get("username")
			messages.success(request, "Account created, you can now log in.")
			return redirect("login")
	else:
		form = UserRegisterForm()
	context = {
		"form": form,
		"title": "Register"
	}
	return render(request, "users/register.html", context)

@login_required
def profile(request):
	if request.method == "POST":
		u_form = UserUpdateForm(request.POST, instance=request.user)
		p_form = ProfileUpdateForm(request.POST, instance=request.user.profile, machine_set=request.user.machine_set)
		if u_form.is_valid() and p_form.is_valid():
			u_form.save()
			p_form.save()
			messages.success(request, f"Changes saved.")
			return redirect("profile")
	else:
		m_set = request.user.machine_set.all()
		for m in m_set:
			if m.marked_for_delete:
				delete_machine(m)

		u_form = UserUpdateForm(instance=request.user)
		p_form = ProfileUpdateForm(instance=request.user.profile.machine, machine_set=request.user.machine_set)

	context = {
		"title": "Profile",
		"u_form": u_form,
		"p_form": p_form,
	}
	return render(request, "users/profile.html", context)