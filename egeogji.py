import random
from random import randrange

class torso():
    
    def __init__(self, mode):
        self.mesh_var = "address to an object that has a cube shape with collider"
        self.joint_types =[ball_socket, hinge]
        self.num_connections = randrange(1,10)
        self.torso = {
            
            "mesh": self.mesh_var,
            "dimensions": [randrange(10,20), randrange(10,20), randrange(10,20)],
            "central_brain": central_brain(self.num_connections),
            "connections": [{
                "limb_object": limb(),
                "position": [randrange(10,20), randrange(10,20), randrange(10,20)],
                "orientation": ["this is a random 3-d vector"],
                "joint_type": random.choice(self.joint_types)
               }
               
               for i in range(self.num_connections)

            ]

    }
        

    def mutate():
        print("this function shud mutate the organism!!!")
        



class brain():
    print("this class defines neuron types and stuff")      

    
      
class central_brain(brain):
    def __init__(self):
        self.sensors = eye.new()

class local_brain(brain):
     def __init__(self):
        pass

class brain():
     def __init__(self):
        pass
     
class eye():
    def __init__(self):
        self.raycast = "this is not a string, it's a pointer to a raycast"

class limb():
    def __init__(self):
        pass


class joint():
    def __init__(self):
        pass


class ball_socket(joint):
    def __init__(self):
        pass


class hinge(joint):
    def __init__(self):
        pass