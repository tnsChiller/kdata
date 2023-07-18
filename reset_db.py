from poker_royale.models import Game, Machine, Training

for i in Game.objects.all(): i.delete()
for i in Machine.objects.all(): i.delete()
for i in Training.objecets.all(): i.delete()
