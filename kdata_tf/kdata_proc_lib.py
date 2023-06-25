from matplotlib import pyplot
import numpy as np

def get_gains(gs,len_modsin):
    stacks = gs.stacks
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
    pyplot.legend(["mod1", "mod2"])
    pyplot.xlabel("Game")
    pyplot.ylabel("Total Stack")
    pyplot.savefig("testfig.png")
    
    return tot_gains

def card_str(card):
    if card % 4 == 0:
        suit = "S"
    elif card % 4 == 1:
        suit = "H"
    elif card % 4 == 2:
        suit = "C"
    else:
        suit = "D"
        
    if card // 4 <= 8:
        number = (card // 4) + 2
    elif card // 4 == 9:
        number = "J"
    elif card // 4 == 10:
        number = "Q"
    elif card // 4 == 11:
        number = "K"
    else:
        number = "A"
    
    return f"{number} - {suit}"

def get_state(gs,idx):
    state = {}
    for player in range(6):
        
        dct = {"stack": gs.stacks[idx,player],
               "move": gs.moves[idx,player],
               "hand_1": card_str(gs.hands[idx,player,0]),
               "hand_2": card_str(gs.hands[idx,player,1])}
        state[f"player_{player}"]= dct
    
    state["mid"] = []
    for i in range(5):
        state["mid"].append(card_str(gs.cards[idx,i]))
    return state