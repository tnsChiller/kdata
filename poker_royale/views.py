from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Game, Machine, Training
from .forms import MachineCreateForm, MachineTrainForm
from kdata_tf.kdata_tf_lib import make_machine
from kdata_tf.kdata_os_lib import update_que
from django.contrib.auth.decorators import login_required
from poker_royale.forms import NewGameForm, NewSparSess
from randomname import get_name
from django.contrib import messages
from django.views.generic import DetailView, ListView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.models import User

game_lifters = 1
train_lifters = 4

class GameDetailView(DetailView):
	model = Game
	def post(self, request, *args, **kwargs):
		pk = kwargs.get("pk")

		if request.method == "POST":
			obj = self.model.objects.filter(pkid=pk).first()
			payload = list(request.POST.keys())[1].split("_")
			if payload[0] == "PLAY":
				print("PLAY RECEIVED")
				obj.m2 = Machine.objects.filter(unid=payload[1]).first()
				obj.status = "QUE"
				obj.save()
				messages.success(request, "Game queued up for execution!")
			elif payload[1] == "CLOSE":
				obj.status = "CLOSE"
				obj.marked_for_close = True
				obj.save()
				messages.warning(request, "Game closed.")
			return render(request, "poker_royale/user_list.html", {"object": obj})


class TrainDetailView(DetailView):
	model = Training
	template_name = "poker_royale/train_detail.html"

# class UserGamesView(DetailView):
# 	model = User
# 	template_name = "poker_royale/user_list.html"
# 	update_que()

@login_required
def UserGamesView(request, pk):
	update_que()

	g_list = []
	m_set = request.user.machine_set.all()
	for m in m_set:
		m1_list = Game.objects.filter(m1=m)
		m2_list = Game.objects.filter(m2=m)

		for g in m1_list:
			if g not in g_list: g_list.append(g)
		for g in m2_list:
			if g not in g_list: g_list.append(g)

	context = {
		"g_list": g_list
	}

	return render(request, "poker_royale/user_list.html",context)

class UserTrainView(DetailView):
	model = User
	template_name = "poker_royale/user_train_list.html"
	update_que()

class GameCloseView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Game
    success_url = "/poker-royale"
    template_name = "poker_royale/close_game.html"

    def test_func(self):
        game = self.get_object()
        if self.request.user == game.creator:
            return True
        return False
        
@login_required
def home(request):
	update_que()
	if request.method == "POST":
		g_form = NewGameForm(request.POST, instance=request.user)
		s_form = NewSparSess(request.POST, instance=request.user)
		kys = list(request.POST.keys())
		if kys[1] == 'entry_cost' and g_form.is_valid():
			num_games = len(Game.objects.filter(creator=request.user, status="QUE"))
			if num_games >= 10:
				messages.warning(request, "You have too many active games, either close some of them or wait for them to be played.")
			else:
				game = Game()
				game.creator = request.user
				game.name = get_name()
				game.entry_cost = g_form.cleaned_data["entry_cost"]
				game.number_of_games = g_form.cleaned_data["number_of_games"]
				game.m1 = request.user.profile.machine
				game.save()
				update_que()
				messages.success(request, f"Game '{game.name}' Created!")
			return redirect("poker-royale-home")

		elif kys[3] == 'm2' and s_form.is_valid():
			num_games = len(Game.objects.filter(creator=request.user,status="QUE"))
			if num_games >= 10:
				messages.warning(request, "You have too many active games, either close some of them or wait for them to be played.")
			else:
				game = Game()
				game.creator = request.user
				game.name = get_name()
				game.entry_cost = 0
				game.m1 = s_form.cleaned_data["m1"]
				game.m2 = s_form.cleaned_data["m2"]
				game.number_of_games = s_form.cleaned_data["number_of_games"]
				game.status = "QUE"
				game.spar = True
				game.save()
				update_que()
				messages.success(request, f"Sparring Session '{game.name}' Created! You can see it in your games.")
			return redirect("poker-royale-home")

	else:
		g_set = request.user.game_set.all()
		for g in g_set:
			if g.marked_for_close:
				g.delete()

		g_form = NewGameForm(instance=request.user)
		s_form = NewSparSess(instance=request.user)
#		s_form.fields["m1"].queryset = Machine.objects.filter(creator=request.user,ready=True)
#		s_form.fields["m2"].queryset = Machine.objects.filter(creator=request.user,ready=True)

	g_list = Game.objects.filter(spar=False, status="WAITING")
	context = {
		"title": "Poker Royale",
		"g_form": g_form,
		"g_list": g_list,
		"s_form": s_form
	}
	return render(request, "poker_royale/home.html", context)

def spectate(request):
	context = {
		"title": "Spectate"
	}
	return HttpResponse('<h1> Watch poker bots in action</h1>', context)

@login_required
def new_machine(request):
	if request.method == "POST":
		form = MachineCreateForm(request.POST)
		if form.is_valid():
			machine = form.save()
			machine.creator = request.user
			make_machine(machine)
			machine.save()
			return redirect("profile")
	else:
		form = MachineCreateForm()
	context = {
		"title": "New Machine",
		"machine_create_form": form
	}
	return render(request, "poker_royale/new_machine.html", context)

@login_required
def train_machine(request):
	if request.method == "POST":
		form = MachineTrainForm(request.POST)
		form.fields["machine"].queryset = Machine.objects.filter(creator=request.user)
		if form.is_valid():
			train = form.save()
			train.save()
			m_obj = train.machine
			m_obj.mark += 1
			m_obj.pk = None
#			m_obj.ready = False
			m_obj.save()
			train.new_pk = m_obj.unid
			train.save()
			update_que()
			return redirect("profile")
	else:
		form = MachineTrainForm()
		form.fields["machine"].queryset = Machine.objects.filter(creator=request.user)
	context = {
		"title": "Train Machine",
		"machine_train_form": form
	}
	return render(request, "poker_royale/train_machine.html", context)

@login_required
def delete_machine(request):
	m_set = request.user.machine_set.all()
	if request.method == "POST":
		for m in m_set:
			if request.POST.get(f"{m.pk}") == "on":
				m.marked_for_delete = True
				m.save()
		return redirect("profile")

	context={
		"machines": m_set
	}

	return render(request, "poker_royale/delete_machine.html", context)

@login_required
def close_games(request):
	g_set = request.user.game_set.all()
	if request.method == "POST":
		for g in g_set:
			if request.POST.get(f"{g.pkid}") == "on":
				g.marked_for_close = True
				g.save()
		return redirect("poker-royale-home")

	context={
		"games": g_set
	}

	return render(request, "poker_royale/close_games.html", context)
