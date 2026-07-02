import random
from random import randrange


class joint():
    def __init__(self):
        pass


class ball_socket(joint):
    def __init__(self):
        pass


class hinge(joint):
    def __init__(self):
        pass


joint_types =[ball_socket, hinge]




class torso():
    def get_new_connection(self):
        pos = ["random position constrained by torso size"]
        dim = ["this has 3 random values in a suitable range"]
        ori = ["this is a random 3-d vector"]
        joint_type = random.choice(joint_types)

        return {
                "position": pos,
                "dimensions": dim,
                "orientation": ori,
                "joint_type": joint_type,
                "limb_object": limb(pos, dim, ori, joint_type)
               }
    

    def __init__(self):
        #this is called when we create a new creature (as one creature can only have one torso)
        self.mesh_var = "address to an object that has a cuboid mesh with collider"
        #mesh_var is basically what we will instantiate using the below parameters
        
        self.min_size = 10
        self.max_size = 20
        self.num_connections = randrange(1,7)
        
        self.torso = {
            "mesh": self.mesh_var,
            "dimensions": [randrange(self.min_size,self.max_size) for i in range(3)],
            "central_brain": central_brain(self.num_connections),
            "connections": [self.get_new_connection() for i in range(self.num_connections)]

        }
        

    def mutate(self, mutation_rate):
        #SAVE THE EXISTING SCHEMA IN A FILE AND THEN MUTATE

        #mutate size
        random_var = randrange(1,100)
        if random_var < mutation_rate:
            self.torso["dimensions"] = [randrange(10,20), randrange(10,20), randrange(10,20)]


        #mutate central brain
        random_var = randrange(1,100)
        if random_var < mutation_rate:
            self.torso["central_brain"].mutate()
        

        #mutate number of limbs
        random_var = randrange(1,100)
        if random_var < mutation_rate:
            new_connections = randrange(1,10)
            if new_connections> self.num_connections:
                for i in range(new_connections)- self.num_connections:
                    self.torso["connections"].append(self.connection_dict)

            elif new_connections<self.num_connections:
                new_rand = randrange(0, self.num_connections)
                connections_list = self.torso["connections"]
                connections_list.pop(new_rand)
                self.torso["connections"] = connections_list

            self.num_connections = new_connections


        #mutate a limb and its local brain
        for i in self.torso["connections"]:
            random_var = randrange(1,100)
            if random_var<mutation_rate:
                pass

        
        


         

        

        

        
        



class brain():
    print("this class defines neuron types and stuff")      

    
      
class central_brain(brain):
    def __init__(self, num_connections):
        self.sensors = eye()

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
    max_depth = 5
    def get_new_connection(self):
        #this has same name as the func in torso, and looks kinda similar. but it is a different function
        pos = ["random position constrained by torso size"]
        dim = ["this has 3 random values in a suitable range"]
        ori = ["this is a random 3-d vector"]
        joint_type = random.choice(joint_types)

        return {
                "position": pos,
                "dimensions": dim,
                "orientation": ori,
                "joint_type": joint_type,
                "limb_object": limb(pos, dim, ori, joint_type, depth= self.depth+1)
               }
    


    def __init__(self, pos, dim, ori, joint_type, depth =1):
        self.depth = depth
        self.has_connection = random.random() < 0.5 and self.depth < self.max_depth
        self.limb = {"pos": pos,
                     "dim": dim,
                     "ori": ori,
                     "joint_type": joint_type,
                     "connection": self.get_new_connection() if self.has_connection else None
                     

                     }




def print_creature():
    new = torso()
    print(f"TORSO  dim={new.torso['dimensions']}")
    for i in new.torso["connections"]:
        print_limb(i, indent=1)

def print_limb(i, indent):
    limb_obj = i["limb_object"]
    l = limb_obj.limb
    joint_name = i["joint_type"].__name__
    prefix = "  " * indent + "└─ "
    print(f"{prefix}LIMB(d{limb_obj.depth}) joint={joint_name}")
    if l["connection"] is not None:
        print_limb(l["connection"], indent + 1)


for i in range(10):
    print_creature()
    print("_______________________________________\n")