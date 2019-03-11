from pyglet.window import key

def key_press(k, mod, action):
    if k==key.LEFT:  action[0] = -1.0
    if k==key.RIGHT: action[0] = +1.0
    if k==key.UP:    action[1] = +1.0
    if k==key.DOWN:  action[2] = +0.8   # set 1.0 for wheels to block to zero rotation

def key_release(k, mod, action):
    if k==key.LEFT  and action[0]==-1.0: action[0] = 0
    if k==key.RIGHT and action[0]==+1.0: action[0] = 0
    if k==key.UP:    action[1] = 0
    if k==key.DOWN: action[2] = 0