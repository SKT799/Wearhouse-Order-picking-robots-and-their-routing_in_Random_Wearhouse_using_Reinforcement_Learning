# Multiagent Routing In Random Envirement Using Dynamic Policy Adoption #

This project solves a problem using Reinforcement Learning(a popular AI technique being used in today's very famous bots and tools like ChatGPT, DeepSeek Reasoning Model, Recomendation Systems etc).  My project not only has solution but I have also simulated the solution in a custom pygame envirenment totally created by me. In the envirenment there are 3 robots that start from any location on the map and their goal is to reach to an assigned goal. They learn to navigate a grid-based envirenment with both static and dynamic obstacles. I have named the static obstacles brics(wall) and moving obstacles humans(which start randomly from a vacant cell in the grid and they move randomly). Our The simulation can be viewed in the video format uploaded in the repo. This simulation mimics real world warehouse or service robot working in hotels avoiding any collision and reaching their goals successfully.

# Features:
1. Multiagents(in my case I have taken 3 agents but it can be in any number) with separate Q-tables and finding best shortest route to reach the assigned goal
2. Dynamic Obstacle avoidence using dynamic policy adoption
3. Assigned 3 Colors to the three goals for three robots
4. How robots go(path) is simulated by drawing line during routing simulation
5. Pygame based 2D grid
6. Parameters according to the real world (no of static obstacles, no of random obstacles, probability of pausing the robot before collision, etc)

# Prerequisites:
python(latest version would be good to go)
pygame
numpy
matplotlib(optional incase you want to see the performance)
*for running the training and simulation run the code-> main__.py
*for seeing how our code performas upon changing the parameters run the code-> Performance_test.py (in the code you can change the parameters)
If you want to know how pygame works then you may watch this- https://youtu.be/AY9MnQ4x3zk?si=i-p6px1lwphqQKj6

# How agents learn to find the best route?
Reinforcement learning is a method used when we don't have data to teach our model how to perform. Suppose a robot inside a lab is built to climb mountains, now we know every mountain is of different shape and randomness, We can never have data to tacle every random situation like what our robot will do if it slips off from any step? in random envirement(and the truth is our world is filled with randomness) RL is strong and robust tool for making AI systems smart enough to take best decisions. There are so many RL learning algorithms like Q-Learning, DQN, Actor-Critic, Monte-Carlo Methods etc. There is two kind of RL learning. First when we have the model of the envirement(like mine) second we don't have the model for learning.
RL is goal based learning i.e. if you know the goal and you want to find a path to reach there and you don't have any data the this method is fantastic which learns by trial and error. For any class of outcomes we will reward our agent or punish it according to the action taken and that's how our agent learns.

In my case, in my 2D grid world I am using model based learning. I have used Q-Learning algorith which you may read here how it works- https://www.geeksforgeeks.org/q-learning-in-python/
Now, in model based RL(here in my grid-world model) I have 3 agents, 3 respective starting points(denoted by home), 3 respective goals, many static obstacles, and many moving obstacles(denoted by humans).
I will randomly initialize the Q-table with zeroes(later it will be updated with learning steps for deriving optimal policy).
Robots have 4 option to move: actions(up,down, left, right) upon taking any action we will move to the next gridcell. Each robot has it's own Q-table so robots are trained sequentially in my case.

For taking an action I have used epsilon-greedy method which is nothing but exploration-exploitation dilemma. In my case exploration probability is εpsilon. read here- https://www.geeksforgeeks.org/machine-learning/epsilon-greedy-algorithm-in-reinforcement-learning/
=> Reward: +100 for reaching the goal, -100 for hitting the static obstacles, -100 for hitting other agents's goal and their starting point, -1 per normal step. If I don't reward -1 per normal step my agent might not choose the shortest path.
=> The Q value is nothing the accumulated reward(or total reward achieved from current state till the goal)
=> Other parameters for Q-learning: learning rate-alpha α(how fast believing the current new values upon taking an action),  discount factor-gamma γ(how much do you believe on the next state's max Q value)

Now follow epsilon-greedy method, to take actions and update Q values when our agent reaches the goal. One episode is process of taking actions and updating Q-values according to the TD update rule from starting point till goal. 100 episodes means we reached goal for 100 times from restarting and updating the Q values.

# Important: I have neglected randomly initialized humans during traing? Why?
Because the humans are randomly initialized(i.e. in any run, they can start from any random vacant location on the map) then they start moving randomly so including them during Q-learning has no meaning or contribution even if I will add them then It will make the learning unstable by adding randomeness in the Q-vales. So how I was able to manage not to collide with humans?
=> by using dynamic policy allocation and randomly pausing the robots from taking action.
I have made a function which utalizes sensor technology(I have assumed I know where the humans are, practically can be seen using sensor by the robots). So now I know where are the humans, I will sometimes pause the robots for a while to let humans move from it's way and some times I will use the awearness of the positions of the humans.
Everytime when robot wants to take action it will calculate it's next state using all 4 possible actions in the grid, now using the laser I know in which cell which things are if any one if found vacant, our agent can safely move there and from that new position it will restart it's policy in this dynamic way to reach the goal. Using some probability I can assign my robot these two methods to avoid the collision.

# Result: Please use the 3 performance png files, and the video for visual understanding
=> As we increse the number of training episodes , accumulated reward increase, after certain very large number of steps like 1000000-steps it get's saturated
=> As we increase number of random humans keeping the pause probability 0, the total action taken by the robots to reach the goal increases
=> As we increase the number of training episodes keeping the random humans constant and low, action taken to reach the goal decreases.

The above three results seems to be obvious for an AI system.

# Application Of my software: Real world senarios
Warehouse robots (like Amazon)
Airport or hospital delivery bots
Hotel smart delivery bots

# Limitations & Future Work

