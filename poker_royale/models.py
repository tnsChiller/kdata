from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator
from randomname import get_name
from uuid import uuid4

class Machine(models.Model):
	name = models.CharField(max_length=30)
	date_created = models.DateTimeField(auto_now_add=True)
	last_trained = models.DateTimeField(default=timezone.now)
	creator = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
	mark = models.IntegerField(default=0)
	unid = models.CharField(primary_key=True, default=uuid4, max_length=255, null=False, editable=False)
	prev_unid = models.CharField(default="0", max_length=255)
	marked_for_delete = models.BooleanField(default=False)

	num_layers = models.IntegerField(default=5,
									 validators=[
						   	         MinValueValidator(1),
						    		 MaxValueValidator(50)
									 ])

	first_layer_neurons = models.IntegerField(default=5,
											  validators=[
										      MinValueValidator(1),
											  MaxValueValidator(1000)
											  ])

	last_layer_neurons = models.IntegerField(default=5,
											 validators=[
								   	         MinValueValidator(1),
								    		 MaxValueValidator(1000)
											 ])

	dropout_period = models.IntegerField(default=51,
										 validators=[
						   	    	     MinValueValidator(1),
						    			 MaxValueValidator(51)
										 ])

	dropout_frequency = models.FloatField(default=0.0,
									      validators=[
						   	    	      MinValueValidator(0.0),
						    	          MaxValueValidator(1.0)
								          ])

	max_norm = models.FloatField(default=99.0,
									 validators=[
						   	         MinValueValidator(0.0),
						    		 MaxValueValidator(99.0)
									 ])

	l1 = models.FloatField(default=0.0,
					  	   validators=[
	   	           	       MinValueValidator(0.0),
	    	       		   MaxValueValidator(99.0)
	    		           ])

	l2 = models.FloatField(default=0.0,
					  	   validators=[
	   	           	       MinValueValidator(0.0),
	    	       		   MaxValueValidator(99.0)
	    		           ])

	ADAM = "adam"
	NADAM = "nadam"
	ADAMAX = "adamax"
	ADAGRAD = "adagrad"
	RMSPROP = "rmsprop"
	SGD = "sgd"
	optimizer_choices = [
		(ADAM,"ADAM"),
		(NADAM,"NADAM"),
		(ADAMAX,"ADAMAX"),
		(ADAGRAD,"ADAGRAD"),
		(RMSPROP,"RMSPROP"),
		(SGD,"SGD"),
	]
	optimizer = models.CharField(choices=optimizer_choices,
								 default=ADAM,
								 max_length=30)

	MSE = "mean_squared_error"
	MAE = "mean_absolute_error"
	MSLE = "mean_squared_logarithmic_error"
	loss_choices = [
		(MSE,"MEAN SQUARED ERROR"),
		(MAE,"MEAN ABSOLUTE ERROR"),
		(MSLE,"MEAN SQUARED LOGARITHMIC ERROR"),
	]
	loss = models.CharField(choices=loss_choices,
							default=MSE,
							max_length=30)

	RELU = "relu"
	SIGMOID = "sigmoid"
	SOFTMAX = "softmax"
	TANH = "tanh"
	SELU = "selu"
	ELU = "elu"
	EXPONENTIAL = "EXPONENTIAL"
	activation_choices = [
		(RELU,"RELU"),
		(SIGMOID,"SIGMOID"),
		(SOFTMAX,"SOFTMAX"),
		(TANH,"TANH"),
		(SELU,"SELU"),
		(ELU,"ELU"),
		(EXPONENTIAL,"exponential"),
	]
	activation = models.CharField(choices=activation_choices,
								  default=RELU,
								  max_length=30)

	RNDN = "random_normal"
	RNDU = "random_uniform"
	TRNN = "truncated_normal"
	ZERO = "zeros"
	ONES = "ones"
	GLON = "glorot_normal"
	GLOU = "glorot_uniform"
	HENO = "he_normal"
	HEUN = "he_uniform"
	IDEN = "identity"
	ORTH = "orthogonal"
	kernel_initializer_choices = [
		(RNDN,"RANDOM NORMAL"),
		(RNDU,"RANDOM UNIFORM"),
		(TRNN,"TRUNCATED NORMAL"),
		(ZERO,"ZEROS"),
		(ONES,"ONES"),
		(GLON,"GLOROT NORMAL"),
		(GLOU,"GLOROT UNIFORM"),
		(HENO,"HE NORMAL"),
		(HEUN,"HE UNIFORM"),
		(IDEN,"IDENTITY"),
		(ORTH,"ORTHOGONAL"),
	]
	kernel_initializer = models.CharField(choices=kernel_initializer_choices,
								  default=RNDN,
								  max_length=30)

	def __str__(self):
		return f"{self.name}_mk{self.mark}"

class Game(models.Model):
	pkid = models.UUIDField(primary_key=True, default=uuid4, editable=False)
	name = models.CharField(max_length=30, default="No-Name")
	m1 = models.ForeignKey(Machine, on_delete=models.CASCADE, null=True, related_name="contender")
	m2 = models.ForeignKey(Machine, on_delete=models.CASCADE, null=True, related_name="challenger")
	max_players = models.IntegerField(default=2,
									  validators=[
								      MinValueValidator(1),
									  MaxValueValidator(6)
									  ])
	num_players = models.IntegerField(default=1)
	entry_cost = models.IntegerField(default=10,
										validators=[
										MinValueValidator(1),
										MaxValueValidator(10000)
										])
	number_of_games = models.IntegerField(default=1000,
											  validators=[
										      MinValueValidator(1),
											  MaxValueValidator(100000)
											  ])
	status = models.CharField(default="WAITING", max_length=30)
	time = models.TimeField(auto_now_add=True)
	final_dif = models.IntegerField(default=0)
	creator = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
	marked_for_close = models.BooleanField(default=False)
	spar = models.BooleanField(default=False)
	plot = models.ImageField(default='default.png', upload_to='lifter-out')

	def __str__(self):
		return self.name

class Training(models.Model):
	pkid = models.UUIDField(primary_key=True, default=uuid4, editable=False)
	time = models.TimeField(auto_now_add=True)
	machine = models.ForeignKey(Machine, on_delete=models.CASCADE, null=False)
	new_pk = models.CharField(default=uuid4, max_length=255)
	num = models.IntegerField(default=1000)
	it = models.IntegerField(default=10)
	n_epochs = models.IntegerField(default=10)
	btc_size = models.IntegerField(default=32)
	learning_rate = models.FloatField(default=0.001)
	shuffle = models.BooleanField(default=True)
	loss = models.FloatField(default=99.0)
	status = models.CharField(default="QUE", max_length=30)

	def __str__(self):
		return f"{self.machine.name} training instance"