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

for running the training and simulation run the code-> main__.py
for seeing how our code performas upon changing the parameters run the code-> Performance_test.py (in the code you can change the parameters)

# How agents learn to find the best route?
