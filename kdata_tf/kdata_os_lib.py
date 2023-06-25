import os
import shutil
import docker
import socket
import pickle
from poker_royale.models import Training, Game, Machine
from users.models import Profile
HOST = "127.0.0.1"
client = docker.from_env()
cont_name = "lifter:1.05_game"
train_cont_lim = 4
game_cont_lim = 1

def update_que():
    client.containers.prune()

    with open("lifter-out/game_dct.pickle", "rb") as f:
        game_dct = pickle.load(f)
        f.close()

    for i in game_dct:
        status = game_dct[i]["status"]
        obj = Game.objects.filter(pkid=i).first()
        if obj and obj.status != "DONE":
            obj.status = status
            if obj.status == "DONE":
                obj.final_dif = game_dct[i]["stack_dif"]
            obj.save()

    with open("lifter-out/train_dct.pickle", "rb") as f:
        train_dct = pickle.load(f)
        f.close()

    for i in train_dct:
        status = train_dct[i]["status"]
        obj = Training.objects.filter(pkid=i).first()
        if obj:
            obj.status = status
            obj.save()

    game_que = Game.objects.filter(status="QUE")
    train_que = Training.objects.filter(status="QUE")
    game_ctr, train_ctr = 0,0
    for game in game_que:
        if game.status == "QUE":
            game_ctr += 1
            if not game.m1: game.m1 = Machine.objects.first()
            if not game.m2: game.m2 = Machine.objects.first()
            game_dct[str(game.pk)] = {"m1_pk": str(game.m1.pk),
                                      "m1_name": str(game.m1.name),
                                      "m1_mark": str(game.m1.mark),
                                      "m2_pk": str(game.m2.pk),
                                      "m2_name": str(game.m2.name),
                                      "m2_mark": str(game.m2.mark),
                                      "num": game.number_of_games,
                                      "status": game.status,
                                      "stack_dif": 0}


    for train in train_que:
        if train.status == "QUE":
            train_ctr += 1
            if not train.machine: train.machine = Machine.objects.first()
            train_dct[str(train.pkid)] = {"m_pk": train.machine.pk,
                                          "new_pk": train.new_pk,
                                          "num": train.num,
                                          "it": train.it,
                                          "n_epochs": train.n_epochs,
                                          "btc_size": train.btc_size,
                                          "learning_rate": train.learning_rate,
                                          "shuffle": train.shuffle,
                                          "loss": train.loss,
                                          "status": train.status,
                                          "fin_loss": 999}

    with open("lifter-out/game_dct.pickle","wb") as f:
        pickle.dump(game_dct,f,pickle.HIGHEST_PROTOCOL)

    with open("lifter-out/train_dct.pickle","wb") as f:
        pickle.dump(train_dct,f,pickle.HIGHEST_PROTOCOL)

    game_cts, train_cts = 0,0
    for cont in client.containers.list():
        if cont.labels["type"]=="game":
            game_cts+=1
        elif cont.labels["type"]=="train":
            train_cts+=1

    train_req = min(train_cont_lim - train_cts, train_ctr)
    game_req = min(1, game_ctr)

    for i in range(train_cts, train_cts + train_req):
        cont = client.containers.run("lifter:1.05_train",
                                     labels={"type": "train"},
                                     detach=True,
                                     volumes={f"{os.getcwd()}/lifter-out": {"bind": "/lifter-out", "mode": "rw"},
                                              f"{os.getcwd()}/kdata_tf/machines": {"bind": "/machines", "mode": "rw"}},
                                     name=f"train_{i+1}")

    if game_req > 0 and game_cts == 0:
        cont = client.containers.run("lifter:1.05_game",
                                     labels={"type": "game"},
                                     detach=True,
                                     volumes={f"{os.getcwd()}/lifter-out": {"bind": "/lifter-out", "mode": "rw"},
                                              f"{os.getcwd()}/kdata_tf/machines": {"bind": "/machines", "mode": "rw"}},
                                     name=f"game_1")



def update_lifters(game_lifters=1, train_lifters=4):
    for cont in client.containers.list():
        for log in cont.logs().decode().splitlines():
            if "--TRAINING DONE--" in log:
                spl = log.split(";")
                new_pk = spl[1]
                loss = float(spl[2])
                cont.kill()
                session = Training.objects.filter(new_pk=new_pk).first()
                session.status = "DONE"
                session.loss = loss
                session.save()
                
            elif "--GAME DONE--" in log:
                spl = log.split(";")
                game_pk = spl[1]
                stack_dif = int(spl[2])
                cont.kill()
                game = Game.objects.filter(pkid=game_pk).first()
                game.status = "DONE"
                game.final_dif = stack_dif
                game.save()
                if stack_dif > 0:
                    winner = Machine.objects.filter(pk = game.m1_pk).first().creator().profile()
                elif stack_dif < 0:
                    winner = Machine.objects.filter(pk = game.m2_pk).first().creator().profile()
                winner.k_money += game.entry_cost*2
                
    client.containers.prune()
    
    game_ctr, train_ctr = 0,0
    for cont in client.containers.list():
        if cont.labels["type"]=="game":
            game_ctr+=1
        elif cont.labels["type"]=="train":
            train_ctr+=1
            
    for i in range(game_ctr,game_lifters):
        cont = client.containers.run("lifter:1.05_game",
                                     ports={"9999/tcp": 4000+i},
                                     labels={"port": str(4000+i),
                                             "type": "game"},
                                     detach=True,
                                     volumes={f"{os.getcwd()}/lifter-out": {"bind": "/lifter-out", "mode": "rw"},
                                              f"{os.getcwd()}/machines": {"bind": "/machines", "mode": "rw"}},
                                     name=f"game_{i+1}")
        
    for i in range(train_ctr,train_lifters):
        cont = client.containers.run("lifter:1.05_train",
                                     ports={"9999/tcp": 4100+i},
                                     labels={"port": str(4100+i),
                                             "type": "train"},
                                     detach=True,
                                     volumes={f"{os.getcwd()}/lifter-out": {"bind": "/lifter-out", "mode": "rw"},
                                              f"{os.getcwd()}/machines": {"bind": "/machines", "mode": "rw"}},
                                     name=f"train_{i+1}")
    
    train_que = Training.objects.filter(status="QUE").order_by("time")
    for session in train_que:
        for cont in client.containers.list():
            print(cont.logs().splitlines())
            c1 = b"--READY--" in cont.logs().splitlines()
            c2 = b"--BUSY--" not in cont.logs().splitlines()
            c3 = cont.labels["type"] == "train"
            c4 = session.status == "QUE"
            if c1 and c2 and c3 and c4:
                old_pk = session.machine.pk
                new_machine = session.machine
                new_machine.mark += 1
                new_machine.pk = None
                new_machine.save()
                data_dct = {
                        "action":"train",
                        "mod_pk": str(old_pk),
                        "new_pk": str(new_machine.pk),
                        "num": str(session.num),
                        "it": str(session.it),
                        "n_epochs": str(session.n_epochs),
                        "btc_size": str(session.btc_size),
                        "learning_rate": str(session.learning_rate),
                        "shuffle": str(int(session.shuffle)),
                    }
                payload = ""
                for i in data_dct:
                    payload += data_dct[i]+"\n"
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((HOST,int(cont.labels["port"])))
                    s.send(payload.encode())
                session.status = "IN PROGRESS"
    
    game_que = Game.objects.filter(status="QUE").order_by("time")
    for session in game_que:
        for cont in client.containers.list():
            c1 = b"--READY--" in cont.logs().splitlines()
            c2 = cont.labels["type"] == "game"
            c3 = session.status == "QUE"
            if c1 and c2 and c3:
                m1 = Machine.objects.filter(pk = session.m1.pk).first()
                m2 = Machine.objects.filter(pk = session.m2.pk).first()
                data_dct = {
                        "action":"play",
                        "m1_pk": str(m1.pk),
                        "m1_name": str(m1.name),
                        "m1_mark": str(m1.mark),
                        "m2_pk": str(m2.pk),
                        "m2_name": str(m2.name),
                        "m2_mark": str(m2.mark),
                        "num": str(session.number_of_games),
                        "pk": str(session.pk)
                    }
                payload = ""
                for i in data_dct:
                    payload += data_dct[i]+"\n"
                    
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((HOST,int(cont.labels["port"])))
                    s.send(payload.encode())
                session.status = "IN PROGRESS"
