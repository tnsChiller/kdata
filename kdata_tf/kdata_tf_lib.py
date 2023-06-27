from tensorflow import keras
import numpy as np
import os
import time
import shutil
import numpy as np
from numpy import logical_and as lnd
from numpy import logical_or as lor
from numpy import logical_not as lno
from matplotlib import pyplot

def getreps(vin):
    reps=np.zeros(vin.shape,np.int64)
    for i in range(7):
        for j in range(7):
            inc=vin[:,:,i]==vin[:,:,j]
            reps[:,:,i]+=inc
    return reps

class gameState():
    turnLim=18
    def __init__(self,num,mods):
        self.num=num
        self.mods=mods
        
        self.deck=np.resize(np.arange(52,dtype=np.int64),(num,52))
        for i in range(num):np.random.shuffle(self.deck[i])
        self.hands=np.stack((self.deck[:,:6],self.deck[:,6:12]),axis=2)
        self.cards=-1*np.ones((num,5),np.int64)
        self.moves=-1*np.ones((num,6),np.int64)
        # self.moves=np.random.randint(0,7,size=(num,6))
        self.bets=np.zeros((num,6),dtype=np.int64)
        self.bets[:,1]=1
        self.bets[:,2]=2
        self.stacks=np.zeros((num,6),np.int64)
        self.stacks[:,1]=-1
        self.stacks[:,2]=-2
        self.turn=3
        self.pot=np.zeros((num,1),np.int64)
        
        self.done=np.zeros((num,1),bool)
        self.done2=np.zeros((num,1),bool)
        self.str=np.zeros((num,6),np.float64)
        self.ctr=np.zeros((num,1),np.int32)
        self.lastBet=2*np.ones((num,1),np.int64)
        self.betRatios=np.resize(np.array([0,4,8,16,32,64]),(num,6))
        
        self.dbst=False
        self.dbsd=False
        self.dbctr=0
        
        self.logbool=False
        self.log=[]
        self.logctr=0
        
    def gamEval(self):
        for i in range(4):
            self.action()
            
            cTurn=self.cards[:,0]!=-1
            cRiver=self.cards[:,3]!=-1
            
            self.cards[:,:3]=self.deck[:,13:16]
            self.cards[:,3]=lno(cTurn)*self.cards[:,3]+cTurn*self.deck[:,17]
            self.cards[:,4]=lno(cRiver)*self.cards[:,4]+cRiver*self.deck[:,19]
        self.showdown()
        
    def action(self):
        for i in range(self.turnLim+6):
            self.step()
        self.ctr=np.zeros((self.num,1),np.int32)
        self.pot[:,0]+=np.sum(self.bets,axis=1)
        self.bets=np.zeros((self.num,6),dtype=np.int64)
        self.turn=1
        for i in range(6):
            c1=self.moves[:,i]==0
            self.moves[:,i]=c1*self.moves[:,i]+lno(c1)*-1*np.ones((self.num),np.int64)
        self.lastBet[:,0]=np.ones((self.num),np.int64)
        
        c0=lor(self.done2[:,0],np.sum(self.moves==0,axis=1)==5)
        self.done[:,0]=c0*self.done[:,0]+lno(c0)*np.zeros((self.num),bool)
        self.done2[:,0]=self.cards[:,4]!=-1
            
    def step(self):
        self.getStensor()
        c0=self.ctr[:,0]>self.turnLim
        self.modout=lno(c0)*self.modout+c0*np.ones(self.modout.shape)
            
        c0=self.moves[:,self.turn]==0
        c1=lor.reduce((c0,self.done[:,0],np.sum(self.moves==0,axis=1)==5))
        self.moves[:,self.turn]=c1*self.moves[:,self.turn]+lno(c1)*self.modout
        
        dif=np.max(self.bets,axis=1)-self.bets[:,self.turn]
        movesWide=np.resize(self.moves[:,self.turn],(7,self.num))
        movesComp=np.transpose(np.resize(np.arange(7),(self.num,7)))
        addBets=np.resize(np.array([dif*0,dif,dif+4,dif+8,dif+16,dif+32,dif+64]),(7,self.num))
        addEncode=(movesWide==movesComp)*addBets
        addSqz=np.sum(addEncode,axis=0)
        self.bets[:,self.turn]=self.done[:,0]*self.bets[:,self.turn]+lno(self.done[:,0])*(self.bets[:,self.turn]+addSqz)
        self.stacks[:,self.turn]=self.done[:,0]*self.stacks[:,self.turn]+lno(self.done[:,0])*(self.stacks[:,self.turn]-addSqz)
        
        c1=self.moves[:,self.turn]<2
        trnv=self.turn*np.ones((self.num),np.int64)
        self.lastBet[:,0]=c1*self.lastBet[:,0]+lno(c1)*trnv
        for i in range(1,6):
            k=(self.turn+i)%6
            c2=lor(c1,self.moves[:,k]==0)
            self.moves[:,k]=c2*self.moves[:,k]+lno(c2)*-1*np.ones((self.num),np.int64)
        
        self.turn=(self.turn+1)%6
        self.done[:,0]=lor.reduce((self.done[:,0],self.turn==self.lastBet[:,0],np.sum(self.moves==0,axis=1)==5))
        self.ctr+=1
    def getStensor(self):
        idx=self.turn%6
        if self.logbool:
            self.logctr+=1
            self.log.append(encodeState(self))
        self.modin=np.concatenate((self.hands[:,idx]//4/12,self.hands[:,idx]%4/4,
                                   self.cards//4/12,self.cards%4/4,
                                   np.concatenate((self.moves[:,idx:6]/6,self.moves[:,0:idx]/6),axis=1),
                                   np.concatenate((self.bets[:,idx:6]/250,self.bets[:,0:idx]/250),axis=1),
                                   np.concatenate((self.stacks[:,idx:6]/2000,self.stacks[:,0:idx]/2000),axis=1),
                                   self.pot/5000),axis=1)
        y=self.mods[idx](self.modin).numpy()
        ymax=np.max(y,axis=1)
        self.modout=((y[:,0]==ymax)*0+(y[:,1]==ymax)*1+(y[:,2]==ymax)*2+
                    (y[:,3]==ymax)*3+(y[:,4]==ymax)*4+(y[:,5]==ymax)*5+
                    (y[:,6]==ymax)*6)
        
    def showdown(self):
        self.getScd()
        cStrf=self.scd['strf']!=-1
        cFoak=self.scd['foak']!=-1
        cFull=lnd(self.scd['toak'][:,:,0]!=-1,self.scd['pair'][:,:,0]!=-1)
        cFlsh=self.scd['flsh'][:,:,0]!=-1
        cStra=self.scd['stra']!=-1
        cToak=self.scd['toak'][:,:,0]!=-1
        cTwop=lnd(self.scd['pair'][:,:,0]!=-1,self.scd['pair'][:,:,1]!=-1)
        cPair=self.scd['pair'][:,:,0]!=-1
        cHigh=self.scd['high']!=-1
        
        cFoak=lnd(cFoak,lno(cStrf))
        cFull=lnd.reduce((cFull,lno(cStrf),lno(cFoak)))
        cFlsh=lnd.reduce((cFlsh,lno(cStrf),lno(cFoak),lno(cFull)))
        cStra=lnd.reduce((cStra,lno(cStrf),lno(cFoak),lno(cFull),lno(cFlsh)))
        cToak=lnd.reduce((cToak,lno(cStrf),lno(cFoak),lno(cFull),lno(cFlsh),lno(cStra)))
        cTwop=lnd.reduce((cTwop,lno(cStrf),lno(cFoak),lno(cFull),lno(cFlsh),lno(cStra),lno(cToak)))
        cPair=lnd.reduce((cPair,lno(cStrf),lno(cFoak),lno(cFull),lno(cFlsh),lno(cStra),lno(cToak),lno(cTwop)))
        
        tmp0=np.maximum(self.scd['toak'][:,:,1],self.scd['high'][:,:,0])
        tmp1=np.maximum(tmp0,self.scd['toak'][:,:,0])
        tmp2=np.maximum(tmp1,self.scd['pair'][:,:,0])
        
        self.str=(cHigh[:,:,4]*13**0*self.scd['high'][:,:,4]+
                  cHigh[:,:,3]*13**1*self.scd['high'][:,:,3]+
                  cHigh[:,:,2]*13**2*self.scd['high'][:,:,2]+
                  cHigh[:,:,1]*13**3*self.scd['high'][:,:,1]+
                  cHigh[:,:,0]*13**4*self.scd['high'][:,:,0]+
                  cPair*13**5*self.scd['high'][:,:,2]+
                  cPair*13**6*self.scd['high'][:,:,1]+
                  cPair*13**7*self.scd['high'][:,:,0]+
                  cPair*13**8*self.scd['pair'][:,:,0]+
                  cTwop*13**9*self.scd['high'][:,:,0]+
                  cTwop*13**10*self.scd['pair'][:,:,1]+
                  cTwop*13**11*self.scd['pair'][:,:,0]+
                  cToak*13**12*tmp0+
                  cToak*13**13*self.scd['toak'][:,:,0]+
                  cStra*13**14*self.scd['stra']+
                  cFlsh*13**15*self.scd['flsh'][:,:,4]+
                  cFlsh*13**16*self.scd['flsh'][:,:,3]+
                  cFlsh*13**17*self.scd['flsh'][:,:,2]+
                  cFlsh*13**18*self.scd['flsh'][:,:,1]+
                  cFlsh*13**19*self.scd['flsh'][:,:,0]+
                  cFull*13**20*self.scd['pair'][:,:,0]+
                  cFull*13**21*self.scd['toak'][:,:,0]+
                  cFoak*13**22*tmp2+
                  cFoak*13**23*self.scd['foak']+
                  cStrf*13**24*self.scd['strf'])
        
        for i in range(6):
            c0=self.moves[:,i]==np.zeros((self.num),np.int64)
            self.str[:,i]=c0*np.zeros((self.num),np.int64)+lno(c0)*self.str[:,i]
            
        winctr=np.zeros(self.num,np.int64)
        maxstr=np.amax(self.str,axis=1)
        for i in range(6):
            c1=np.abs(1-self.str[:,i]/maxstr)<1e-27
            winctr+=c1
        
        amt=self.pot[:,0]//winctr
        for i in range(6):
            c2=self.str[:,i]==maxstr
            self.pot[:,0]=lno(c2)*self.pot[:,0]+c2*(self.pot[:,0]-amt)
            self.stacks[:,i]=lno(c2)*self.stacks[:,i]+c2*(self.stacks[:,i]+amt)
            
        if self.dbsd:
            print("Winner: {}".format(np.where(self.str==np.max(self.str))[1]))
            print("Nos: ")
            print(self.sev//4)
            print("Suits: ")
            print(self.sev%4)
    
            cStrf=self.scd['strf']!=-1
            cFoak=self.scd['foak']!=-1
            cFull=lnd(self.scd['toak'][:,:,0]!=-1,self.scd['pair'][:,:,0]!=-1)
            cFlsh=self.scd['flsh'][:,:,0]!=-1
            cStra=self.scd['stra']!=-1
            cToak=self.scd['toak'][:,:,0]!=-1
            cTwop=lnd(self.scd['pair'][:,:,0]!=-1,self.scd['pair'][:,:,1]!=-1)
            cPair=self.scd['pair'][:,:,0]!=-1
            cHigh=self.scd['high']!=-1
            
            for i in range(6):
                if cStrf[0,i]:
                    print("P{}: cStrf".format(i))
                elif cFoak[0,i]:
                    print("P{}: cFoak".format(i))
                elif cFull[0,i]:
                    print("P{}: cFull".format(i))
                elif cFlsh[0,i]:
                    print("P{}: cFlsh".format(i))
                elif cStra[0,i]:
                    print("P{}: cStra".format(i))
                elif cToak[0,i]:
                    print("P{}: cToak".format(i))
                elif cTwop[0,i]:
                    print("P{}: cTwop".format(i))
                elif cPair[0,i]:
                    print("P{}: cPair".format(i))
                else:
                    print("P{}: high".format(i))
            print("-----------------------------\n")

    def getScd(self):
        scd={}
        scd['strf']=-1*np.ones((self.num,6),np.int64)
        scd['foak']=-1*np.ones((self.num,6),np.int64)
        scd['flsh']=-1*np.ones((self.num,6,5),np.int64)
        scd['stra']=-1*np.ones((self.num,6),np.int64)
        scd['toak']=-1*np.ones((self.num,6,2),np.int64)
        scd['pair']=-1*np.ones((self.num,6,3),np.int64)
        scd['high']=-1*np.ones((self.num,6,5),np.int64)
        
        sev=np.zeros((self.num,6,7),np.int64)
        sev[:,:,:2]=self.hands
        sev[:,:,2:5]=np.resize(np.repeat(self.deck[:,13:16],6,axis=0),(self.num,6,3))
        sev[:,:,5]=np.resize(np.repeat(self.deck[:,17],6),(self.num,6))
        sev[:,:,6]=np.resize(np.repeat(self.deck[:,19],6),(self.num,6))
        
        sev.sort(axis=2)
        nos=sev//4
        sts=sev%4
        rnos=getreps(nos)        
        rsts=getreps(sts)
        
        for i in range(7):
            cFoak=rnos[:,:,i]==4
            cFlsh=rsts[:,:,i]==5
            cToak=rnos[:,:,i]==3
            cPair=rnos[:,:,i]==2
            cHigh=rnos[:,:,i]==1
            
            
            c0=lnd(cFlsh,scd['flsh'][:,:,0]<nos[:,:,i])
            for j in range(4):
                scd['flsh'][:,:,4-j]=lno(c0)*scd['flsh'][:,:,4-j]+c0*scd['flsh'][:,:,3-j]
            scd['flsh'][:,:,0]=lno(c0)*scd['flsh'][:,:,0]+c0*nos[:,:,i]
            
            c0=lnd(cToak,scd['toak'][:,:,0]<nos[:,:,i])
            scd['toak'][:,:,1]=lno(c0)*scd['toak'][:,:,1]+c0*scd['toak'][:,:,0]
            scd['toak'][:,:,0]=lno(c0)*scd['toak'][:,:,0]+c0*nos[:,:,i]
            
            c0=lnd(cPair,scd['pair'][:,:,0]<nos[:,:,i])
            for j in range(2):
                scd['pair'][:,:,2-j]=lno(c0)*scd['pair'][:,:,2-j]+c0*scd['pair'][:,:,1-j]
            scd['pair'][:,:,0]=lno(c0)*scd['pair'][:,:,0]+c0*nos[:,:,i]
            
            c0=lnd(cHigh,scd['high'][:,:,0]<nos[:,:,i])
            for j in range(4):
                scd['high'][:,:,4-j]=lno(c0)*scd['high'][:,:,4-j]+c0*scd['high'][:,:,3-j]
            scd['high'][:,:,0]=lno(c0)*scd['high'][:,:,0]+c0*nos[:,:,i]
            
            c0=lnd(cFoak,scd['foak']<nos[:,:,i])
            scd['foak']=lno(c0)*scd['foak']+c0*nos[:,:,i]
        
        cStra=np.zeros((self.num,6,4),bool)
        cStrf=np.zeros((self.num,6,4),bool)
        cAce=nos[:,:,-1]==12
        tmp=lno(cAce)*-10*np.ones((self.num,6),np.int64)+cAce*np.zeros((self.num,6),np.int64)
        nos+=1
        nos=np.concatenate((np.resize(tmp,(self.num,6,1)),nos),axis=2)
        sts=np.concatenate((np.resize(sts[:,:,-1],(self.num,6,1)),sts),axis=2)
        for i in range(4):
            for j in range(i+1,8):
                cs0=nos[:,:,i]==nos[:,:,j]-1
                cs1=nos[:,:,i]==nos[:,:,j]-2
                cs2=nos[:,:,i]==nos[:,:,j]-3
                cs3=nos[:,:,i]==nos[:,:,j]-4
                
                csf0=lnd(cs0,sts[:,:,i]==sts[:,:,j])
                csf1=lnd(cs1,sts[:,:,i]==sts[:,:,j])
                csf2=lnd(cs2,sts[:,:,i]==sts[:,:,j])
                csf3=lnd(cs3,sts[:,:,i]==sts[:,:,j])
                
                cStra[:,:,0]=lor(cStra[:,:,0],cs0)
                cStra[:,:,1]=lor(cStra[:,:,1],cs1)
                cStra[:,:,2]=lor(cStra[:,:,2],cs2)
                cStra[:,:,3]=lor(cStra[:,:,3],cs3)
                
                cStrf[:,:,0]=lor(cStrf[:,:,0],csf0)
                cStrf[:,:,1]=lor(cStrf[:,:,1],csf1)
                cStrf[:,:,2]=lor(cStrf[:,:,2],csf2)
                cStrf[:,:,3]=lor(cStrf[:,:,3],csf3)
                
            c0=np.all(cStra,axis=2)
            c1=np.all(cStrf,axis=2)
            scd['stra']=lno(c0)*scd['stra']+c0*nos[:,:,i]
            scd['strf']=lno(c1)*scd['strf']+c1*nos[:,:,i]
            cStra=np.zeros((self.num,6,4),bool)
            cStrf=np.zeros((self.num,6,4),bool)
        self.scd=scd
        
def encodeState(gs):
    out=np.zeros((gs.num,59),np.int64)
    out[:,:20]=gs.deck[:,:20]
    tmp=gs.hands[:,:,0]
    out[:,20:26]=np.concatenate((tmp[:,gs.turn:6],tmp[:,0:gs.turn]),axis=1)
    tmp=gs.hands[:,:,1]
    out[:,26:32]=np.concatenate((tmp[:,gs.turn:6],tmp[:,0:gs.turn]),axis=1)
    out[:,32:37]=gs.cards
    tmp=gs.stacks
    out[:,37:43]=np.concatenate((tmp[:,gs.turn:6],tmp[:,0:gs.turn]),axis=1)
    tmp=gs.bets
    out[:,43:49]=np.concatenate((tmp[:,gs.turn:6],tmp[:,0:gs.turn]),axis=1)
    tmp=gs.moves
    out[:,49:55]=np.concatenate((tmp[:,gs.turn:6],tmp[:,0:gs.turn]),axis=1)
    out[:,55]=gs.pot[:,0]
    out[:,56]=gs.done[:,0]
    out[:,57]=(gs.lastBet[:,0]-gs.turn)%6
    out[:,58]=gs.ctr[:,0]
    return out

def decodeState(egs):
    out=gameState(egs.shape[0],[])
    out.deck[:,:20]=egs[:,:20]
    out.hands[:,:,0]=egs[:,20:26]
    out.hands[:,:,1]=egs[:,26:32]
    out.cards=egs[:,32:37]
    out.stacks=egs[:,37:43]
    out.bets=egs[:,43:49]
    out.moves=egs[:,49:55]
    out.pot[:,0]=egs[:,55]
    out.done[:,0]=egs[:,56]
    out.lastBet[:,0]=egs[:,57]
    out.ctr[:,0]=egs[:,58]
    out.turn=0
    return out

def getModin(egs0):
    hands=np.zeros((egs0.shape[0],6,2),np.int64)
    hands[:,:,0]=egs0[:,20:26]
    hands[:,:,1]=egs0[:,26:32]
    cards=egs0[:,32:37]
    bets=egs0[:,43:49]
    moves=egs0[:,49:55]
    stacks=egs0[:,37:43]
    pot=np.zeros((egs0.shape[0],1))
    pot[:,0]=egs0[:,55]
    
    out=np.concatenate((hands[:,0]//4/12,hands[:,0]%4/4,
                           cards//4/12,cards%4/4,
                           np.concatenate((moves[:,0:6]/6,moves[:,0:0]/6),axis=1),
                           np.concatenate((bets[:,0:6]/250,bets[:,0:0]/250),axis=1),
                           np.concatenate((stacks[:,0:6]/2000,stacks[:,0:0]/2000),axis=1),
                           pot/5000),axis=1,dtype=np.float32)
    return out

def make_machine(machine):
    num_layers = machine.num_layers
    first_layer_neurons = machine.first_layer_neurons
    last_layer_neurons = machine.last_layer_neurons
    dropout_period = machine.dropout_period
    dropout_frequency = machine.dropout_frequency
    max_norm = machine.max_norm
    l1 = machine.l1
    l2 = machine.l2
    optimizer = machine.optimizer
    loss = machine.loss
    activation = machine.activation
    kernel_initializer = machine.kernel_initializer

    neurons=np.linspace(first_layer_neurons,last_layer_neurons,num_layers,dtype=np.int32)

    if optimizer=="adam":
        opt=keras.optimizers.Adam()
    elif optimizer=="nadam":
        opt=keras.optimizers.Nadam()
    elif optimizer=="adamax":
        opt=keras.optimizers.Adamax()
    elif optimizer=="adagrad":
        opt=keras.optimizers.Adagrad()
    elif optimizer=="rmsprop":
        opt=keras.optimizers.RMSprop()
    else:
        opt=keras.optimizers.SGD()

    model=keras.models.Sequential()
    layers=[keras.layers.Input(shape=(33))]

    ctr=1
    for i in range(len(neurons)):
        layers.append(keras.layers.Dense(neurons[i],activation=activation,
                               kernel_initializer=kernel_initializer,
                                kernel_constraint=keras.constraints.max_norm(max_norm,axis=0),
                                kernel_regularizer=keras.regularizers.l1_l2(l1,l2)))
        if ctr%dropout_period==0:layers.append(keras.layers.Dropout(dropout_frequency))
        ctr+=1
    layers.append(keras.layers.Dense(7,activation='tanh'))
    for i in layers:model.add(i)

    model.compile(loss=loss,optimizer=opt,metrics=[])
    model.save(f"kdata_tf/machines/{machine.pk}.h5")

def delete_machine(mach):
    machine_path = f"kdata_tf/machines/{mach.pk}.h5"
    if os.path.isfile(machine_path):
        os.remove(machine_path)
        mach.delete()
    else:
        print(f"machine with id {mach.pk} doesn't exist, deleting database entry...")
        mach.delete()

def generate_data(num,mod):
    mods=[mod for i in range(6)]
    gs=gameState(num,mods)
    gs.logbool=True
    print(f"Number of games: {num}")
    t0=time.perf_counter()
    
    gs.gamEval()
    t1=time.perf_counter()
    print(f"Base games: DONE ({round(t1-t0,4)} s)")
    
    denseLog=np.concatenate(gs.log,axis=0)
    del(gs)
    c0=denseLog[:,49]==0
    c1=np.sum(denseLog[:,49:55]==0,axis=1)==5
    c2=np.logical_not(np.logical_or(c0,c1))
    dlen=len(c1)
    prunedLog=[]
    for i in range(dlen):
        if c2[i]:prunedLog.append(denseLog[i])
    del(denseLog,c0,c1,c2)
    prunedLog=np.stack(prunedLog,axis=0)
    t2=time.perf_counter()
    print(f"Processing base game data: DONE ({round(t2-t1,4)} s)")
    
    branchLog=[]
    tmp=prunedLog.copy()
    dlen=tmp.shape[0]
    for i in range(7):
        tmp[:,49]=i
        branchLog.append(tmp.copy())
    branchLog=np.concatenate(branchLog,axis=0)
    
    newState=decodeState(branchLog)
    del(branchLog)
    newState.mods=[mod for i in range(6)]
    newState.gamEval()
    yFlat=np.tanh(newState.stacks[:,0]/2000)
    t3=time.perf_counter()
    print(f"Event tree analysis: DONE ({round(t3-t2,4)}) s")
    
    x=getModin(prunedLog)
    y=np.stack([yFlat[i*dlen:(i+1)*dlen] for i in range(7)],axis=1)

    return x,y

def train_machine(mod_pk, new_pk, num,it, n_epochs,learning_rate, btc_size, shuffle):
    mod = keras.models.load_model(f"machines/{mod_pk}.h5")
    mod.optimizer.learning_rate = learning_rate
    x,y = generate_data(num,mod)
    for i in range(it):
        t0 = time.perf_counter()
        train_hist = mod.fit(x,y,
                             shuffle=shuffle,
                             epochs=n_epochs,
                             batch_size=btc_size,
                             use_multiprocessing=True,
                             workers=10,
                             verbose=1)
        t1 = time.perf_counter()
        print(f"Training machine: iteration {i+1}/{it} complete. ({round(t1-t0,3)} s")
    mod.save(f"{os.getcwd()}/machines/{new_pk}.h5")
    return round(train_hist.history['loss'][-1],8)

def play_games(num, mod_pks, mod_names, game_pk):
    modsin = []
    for pk in mod_pks:
        modsin.append(keras.models.load_model(f"{pk}.h5"))
    mods=[]
    for i in range(6):
        idx = i % len(modsin)
        mods.append(modsin[idx])
        idx += 1
    gs = gameState(num, mods)
    t0 = time.perf_counter()
    gs.gamEval()
    t1 = time.perf_counter()
    print(f"{num} games played, time = {t1 - t0}")
    stacks = gs.stacks
    len_modsin=len(modsin)
    if len_modsin == 2:
        m_pos = [[0, 2, 4], [1, 3, 5]]
    elif len_modsin == 3:
        m_pos = [[0, 3], [1, 4], [2, 5]]
    else:
        m_pos = [[0], [1], [2], [3], [4], [5]]
    gains=[]
    for m in m_pos:
        m_gains = [stacks[:,idx] for idx in m]
        gains.append(sum(m_gains))
    gains = np.array(gains)
    tot_gains = np.zeros(gains.shape)
    tot_gains[:,0] = gains[:,0]
    for i in range(1,tot_gains.shape[1]):
        tot_gains[:,i] = tot_gains[:,i-1] + gains[:,i]
        
    pyplot.figure(figsize=(35,20))
    for i in range(tot_gains.shape[0]):
        pyplot.plot(tot_gains[i])
        pyplot.plot(tot_gains[i])
    pyplot.legend(mod_names)
    pyplot.xlabel("Game Number")
    pyplot.ylabel("Total Stack")
    pyplot.savefig(f"{os.getcwd()}/lifter-out/{game_pk}.png")
    return tot_gains[0,-1] - tot_gains[1,-1]
