from django.urls import path
from . import views
from .views import (GameDetailView,
                    UserGamesView,
                    GameCloseView,
                    UserTrainView,
                    TrainDetailView)

urlpatterns = [
    path("", views.home, name='poker-royale-home'),
    path("new-machine/", views.new_machine, name='new-machine'),
    path("train-machine/", views.train_machine, name='train-machine'),
    path("delete-machine/", views.delete_machine, name='delete-machine'),
    path("spectate", views.spectate, name='spectate'),
    path("close-games", views.close_games, name='close-games'),
    path("game/<pk>/", GameDetailView.as_view(), name='game-detail'),
    path("user/<pk>/", UserGamesView, name='user-games'),
    path("game/<pk>/close", GameCloseView.as_view(), name='game-close'),
    path("usertrain/<pk>/", UserTrainView.as_view(), name='user-trains'),
    path("train/<pk>/", TrainDetailView.as_view(), name='train-detail'),
]