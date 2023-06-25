from django.db import models
from django.contrib.auth.models import User
from poker_royale.models import Machine

class Profile(models.Model):
	user = models.OneToOneField(User, on_delete=models.CASCADE)
	machine = models.ForeignKey(Machine, on_delete=models.SET_NULL, null=True)
	k_money = models.IntegerField(default=0)

	def __str__(self):
		return f"{self.user.username} Profile"