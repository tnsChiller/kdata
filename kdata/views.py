from django.shortcuts import render

def home(request):
	context = {
		"title": "Home"
	}
	return render(request, "kdata/home.html", context)