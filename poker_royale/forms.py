from django import forms
from .models import Machine, Training, Game
from bootstrap_modal_forms.forms import BSModalModelForm

class MachineCreateForm(forms.ModelForm):
	class Meta:
		model = Machine
		fields = ["name",
				"num_layers",
				"first_layer_neurons",
				"last_layer_neurons",
				"dropout_period",
				"dropout_frequency",
				"max_norm",
				"l1",
				"l2",
				"optimizer",
				"loss",
				"activation",
				"kernel_initializer"]

class MachineTrainForm(forms.ModelForm):
	class Meta:
		model = Training
		fields = ["machine",
				"num",
				"it",
				"n_epochs",
				"btc_size",
				"learning_rate",
				"shuffle"]
				
class NewGameForm(forms.ModelForm):
	class Meta:
		model = Game
		fields = ["entry_cost",
				  "number_of_games"]

class NewSparSess(forms.ModelForm):
	class Meta:
		model = Game
		fields = ["number_of_games",
				  "m1",
				  "m2"]