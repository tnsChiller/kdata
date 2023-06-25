import pickle
from kdata_tf_lib import train_machine
import time
import os

with open("lifter-out/train_dct.pickle", "rb") as f:
	dct = pickle.load(f)
	f.close()

	for i in dct:
		if dct[i]["status"] == "QUE":
			key = i
			dct[key]["status"] = "IN PROGRESS"
			session = dct[key]
			break
		else:
			key = -1

with open("lifter-out/train_dct.pickle", "wb") as f:		
	pickle.dump(dct,f,pickle.HIGHEST_PROTOCOL)
	f.close()

if key != -1:
	print(f"{session}")
	fin_loss = train_machine(mod_pk=session["m_pk"],
							  new_pk=session["new_pk"],
							  num=session["num"],
							  it=session["it"],
							  n_epochs=session["n_epochs"],
							  learning_rate=session["learning_rate"],
							  btc_size=session["btc_size"],
							  shuffle=session["shuffle"])

	dct[key]["fin_loss"] = fin_loss
	dct[key]["status"] = "DONE"

	with open("lifter-out/train_dct.pickle", "wb") as f:
		pickle.dump(dct,f,pickle.HIGHEST_PROTOCOL)
		f.close()