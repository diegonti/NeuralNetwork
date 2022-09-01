import random, pygame

pygame.init()
clock = pygame.time.Clock()

w,h = 600,600
screen = pygame.display.set_mode((w,h))
pygame.display.set_caption("Perceptron")

def sign(n):
    if n >= 0: return 1
    if n < 0: return -1

def map_range(value, a1, b1, a2, b2):
    #To map some value from a range to another range ((-1,1) to pixels)
   return (value - a1) / (b1 - a1) * (b2 - a2) + a2

def f(x):
    #generic line to segment screen
    return -0.3*x-0.2 #small numbers because x,y are -1,1

# NN works with a set of inputs, a hidden layer that is "the brain" witch
# process the information and with an activation/guess function (wheigted sum)
# it gives a numerical output
class Perceptron():

    #Initialyze weitghts randomly

    def __init__(self):
        self.weights = [random.uniform(-1,1) for i in range(3)]
        self.lr = 0.0005 #learning rate to training
        global weights, lr
        weights, lr = self.weights, self.lr

    def guess(self, inputs): #Activation function
        sum = 0 #Weighted sum of imputs and weights
        for i in range(len(weights)):
            sum += inputs[i] * weights[i]
        output = sign(sum)
        return output #Returs output of guess (-1 or 1)

    def train(self, inputs, target): #Recives inputs and the desired answer (known)
        prediction = self.guess(inputs)
        error = target -  prediction
        for i in range(len(weights)): #Tweaking wheigts with ponderated error
            weights[i] += error * inputs[i] * lr

    def guess_line(self,x): #line that perceptrion thinks from its weights and the sum
        m = -weights[0]/weights[1]
        b = -weights[2]/weights[1]
        return  m * x + b


class Point:

    def __init__(self): #Indicating point atributes (x,y,label)
        self.x = random.uniform(-1,1)
        self.y = random.uniform(-1,1)
        self.b = 1
        if self.y > f(self.x): self.label = 1
        elif self.y <= f(self.x): self.label = -1

    def pixelX(self):
        return map_range(self.x,-1,1,0,w)

    def pixelY(self):
        return map_range(self.y,-1,1,h,0)

    def draw(self): #drawing method for avobe or below line (y=-x)
        if self.label == 1: color = (0,0,0)
        elif self.label == -1: color = (0,0,255) 
        px,py = self.pixelX(),self.pixelY()
        pygame.draw.circle(screen, color, (px,py), 5)


perceptron = Perceptron()

nPoints = 200
points = []
for i in range(nPoints):
    points.append(Point())
    

running = True
while running:
    screen.fill((255,255,255))
    px1,px2 = map_range(-1,-1,1,0,w),map_range(1,-1,1,0,w) 
    py1,py2 = map_range(f(-1),-1,1,h,0),map_range(f(1),-1,1,h,0) 
    pygame.draw.line(screen,(0,0,0), (px1,py1),(px2,py2),2)



    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for p in points: #Drawing random points (training dataset)
        inputs = p.x,p.y,p.b
        target = p.label
        
        p.draw()
        guess = perceptron.guess(inputs)
        if guess == target: color = (0,255,0)
        elif guess != target: color = (255,0,0)            
        pygame.draw.circle(screen, color, (p.pixelX(),p.pixelY()), 5)

        perceptron.train(inputs,target) #tweak weights
 
    py3 = map_range(perceptron.guess_line(-1),-1,1,h,0)
    py4 = map_range(perceptron.guess_line(1),-1,1,h,0) 
    pygame.draw.line(screen,(0,0,0), (px1,py3),(px2,py4),1)

    pygame.display.update()
    clock.tick(10) #sets the while cycle frame rate
