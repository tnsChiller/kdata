import pickle
from kdata_tf_lib import play_games
import time

with open("lifter-out/game_dct.pickle", "rb") as f:
	dct = pickle.load(f)
	f.close()

session_dct = {}
for i in dct:
	if dct[i]["status"] == "QUE" and len(session_dct) < 10:
		dct[i]["status"] = "IN PROGRESS"
		session_dct[i] = dct[i]

with open("lifter-out/game_dct.pickle", "wb") as f:
	pickle.dump(dct,f,pickle.HIGHEST_PROTOCOL)
	f.close()



if len(session_dct) > 0:
	for key in session_dct:
		session = session_dct[key]
		stack_dif = play_games(session["num"],
							   [session["m1_pk"],session["m2_pk"]],
                               [f"{session['m1_name']}_{session['m1_mark']}",
                                f"{session['m2_name']}_{session['m2_mark']}"],
							   key)
		dct[key]["stack_dif"] = stack_dif
		dct[key]["status"] = "DONE"

with open("lifter-out/game_dct.pickle", "wb") as f:
	pickle.dump(dct,f,pickle.HIGHEST_PROTOCOL)
	f.close()

print("TEST PRINT")
