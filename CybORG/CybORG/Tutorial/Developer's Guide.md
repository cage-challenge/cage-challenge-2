# Cyborg Developer's Guide
Version 2.0.0
## Installation Instructions
We recommend using a virtual environment running python 3.8 or later. The code below has been tested using an Anaconda virtual environment running python 3.8.11.

Clone the repo using git.
```
git clone https://github.com/cage-challenge/cage-challenge-1.git
```

Install using pip.

```
pip install -e cage-challenge-1/CybORG
```

Confirm system is installed correctly by running tests.
```
 pytest cage-challenge-1/CybORG/CybORG/Tests/test_sim
```
## CybORG in Context
CybORG is a platform designed to assist with the research and development of autonomous network defenders. It allows for the simulation of several cybersecurity scenarios in which an autonomous adversary attempts to compromise a network, while an autonomous network defender tries to stop them.

Our system is designed to train agents via reinforcement learning. This paradigm sees an agent learn by interacting with an environment and receiving feedback on it's actions. Over time, the agent (hopefully) learns which actions are 'good' and which actions are 'bad' in any given context.

One of the most popular environments for reinforcement learning is OpenAI Gym. This is a collection of environments which all share a common API. CybORG models it's API off OpenAI gym, with some small modifications.

The following code shows how an agent typically interacts with CybORG:
```
import inspect
from CybORG import CybORG
from CybORG.Agents import B_lineAgent

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
cyborg = CybORG(path)

results = env.reset(agent='Red')
agent = B_lineAgent()

for step in range(30):
    action = agent.get_action(results.observation,results.action_space)
    results = cyborg.step(action=action,agent='Red')
    print(results.reward) 

```
In the above code, we instantiate the CybORG class by specifying the path to the desired scenario file. We then call the reset method, which instantiates the scenario and provides the initial data for the agent bundled into a custom Results object. For demonstration purposes we are using a pre-made attacking agent, commonly known as 'red-team' in a cybersecurity context. The agent parameter 'Red' in the reset method thus means we want the initial observation for the red team. We then instantiate the agent class. Some agents require input from the initial results object, so this data could be passed into the agent constructor here.

The scenario begins inside the for-loop, where we have decided it will run for 30 steps. For every step, we get an action from the agent and pass that into CybORG via the step method. Again we need to specify that it is red team who is taking the action. We then get a new results object, which will be passed to the agent in the next iteration of the loop. Just for demonstration purposes, we print the reward attribute of the results object. This number is what is used as feedback for the agent, although it isn't being used here.

## The CybORG Class
The CybORG class that we imported in the previous example is defined in the main CybORG directory in the CybORG.py file. The function of this class is to provide an external facing api as well as instantiate the environment controller, which does all of the real work.

CybORG is designed to allow for both simulation and emulation environments under the hood, so one one hand this class acts as a factory choosing which type of environmental controller to instantiate. This is determined by the environment parameter in the class constructor, which defaults to simulation. This guide will focus on the CybORG simulator only, so we will assume the default parameter here always applies.

The key API methods are those called in our example: step and reset. Everything else is some sort of debugging tool to help the researcher see the internal state of the network. Both types of methods delegate everything to the environment controller.
## The Simulation Controller
Because we are assuming we are in simulation mode, the environment controller we want is the SimulationController class found in Simulation/SimulationController.py. However, this class also inherits functionality from the EnvironmentController in Shared/EnvironmentController.py. The SimulationController's main purpose is to manage the internal state as well as the various internal and external agents.

The SimulationController is instantiated by the CybORG class constructor. It passes the scenario file path to the SimulationController class constructor, which parses the file and instantiates a State object to represent the current state of the simulated network. The class constructor also instantiates the internal agents, which are given their own AgentInterface objects. This part of the code is inherited from the Environment Controller.

The two crucial methods, reset and step, are also mostly inherited from the EnvironmentController. The reset method returns the State object and internal Agents to their initial configuration before sending the initial results object to the CybORG class to give to the external agent.

Meanwhile, the workhorse of this class is the step method, which is passed the action provided by the external agent from the CybORG class. The method iterates over each of the agents defined by the scenario (usually Red, Blue and Green). If the agent's action has been provided externally the action is checked to make sure it is valid before it is executed. If the agent's action has not been provided exernally, the internal agent is queried instead to provide its action and this is checked and executed instead.

The executed action returns an Observation object, which shall help the agent work out how to next make it's next decision. The Blue agent then executes a special Monitor action to update its observation so it can see what the other agents have been up to. The method then checks to see if the scenario is finished (the done signal), before computing the reward for each agent, which shall be used to evaluate how well each agent is performing. This data is bundled into a results object and returned to the user.
## The State Object
The State object can be found in HERE and represents the internal state of the simulated network. It is instantiated by THIS.
## Scenario Files
The scenario files are found in
yaml
structure
Image files
## The Host Class
## Red Actions
