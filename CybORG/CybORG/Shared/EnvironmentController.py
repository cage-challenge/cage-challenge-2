## The following code contains work of the United States Government and is not subject to domestic copyright protection under 17 USC § 105.
## Additionally, we waive copyright and related rights in the utilized code worldwide through the CC0 1.0 Universal public domain dedication.

import sys
import yaml
from copy import deepcopy
from prettytable import PrettyTable
from CybORG.Shared import Scenario
from CybORG.Shared.Actions.Action import Sleep, InvalidAction
from CybORG.Shared.Enums import FileType, OperatingSystemType
from CybORG.Shared.Results import Results
from CybORG.Shared.Observation import Observation
from CybORG.Shared.Actions import Action, FindFlag, Monitor
from CybORG.Shared.AgentInterface import AgentInterface
import CybORG.Agents

import numpy as np


from copy import deepcopy
from prettytable import PrettyTable
import numpy as np

class Afterstate():
    def __init__(self,init_state):
        self.blue_info = {}
        self._process_initial_obs(init_state)

    def reset(self,init_state):   
        self.blue_info = {}     
        self._process_initial_obs(init_state)
        self.observation_change(init_state,baseline=True)

    def get_table(self,output_mode='blue_table'):
        if output_mode == 'blue_table':
            return self._create_blue_table(success=None)
        elif output_mode == 'true_table':
            return self.env.get_table()

    def observation_change(self,observation,baseline=False):
        obs = observation if type(observation) == dict else observation.data
        obs = deepcopy(observation)
        success = obs['success']

        self._process_last_action()
        anomaly_obs = self._detect_anomalies(obs) if not baseline else obs
        del obs['success']
        # TODO check what info is for baseline
        info = self._process_anomalies(anomaly_obs)
        if baseline:
            for host in info:
                info[host][-2] = 'None'
                info[host][-1] = 'No'
                self.blue_info[host][-1] = 'No'

        self.info = info

        return self._create_vector(success)


    def _process_initial_obs(self, obs):
        obs = obs.copy()
        self.baseline = obs
        del self.baseline['success']
        for hostid in obs:
            if hostid == 'success':
                continue
            host = obs[hostid]
            interface = host['Interface'][0]
            subnet = interface['Subnet']
            ip = str(interface['IP Address'])
            hostname = host['System info']['Hostname']
            self.blue_info[hostname] = [str(subnet),str(ip),hostname, 'None','No']
        return self.blue_info

    def _process_last_action(self):
        action = self.last_action
        if action is not None:
            name = action.__class__.__name__
            hostname = action.get_params()['hostname'] if name in ('Restore','Remove') else None

            if name == 'Restore':
                self.blue_info[hostname][-1] = 'No'
            elif name == 'Remove':
                compromised = self.blue_info[hostname][-1]
                if compromised != 'No':
                    self.blue_info[hostname][-1] = 'Unknown'

    def _detect_anomalies(self,obs):
        if self.baseline is None:
            raise TypeError('BlueTableWrapper was unable to establish baseline. This usually means the environment was not reset before calling the step method.')

        anomaly_dict = {}

        for hostid,host in obs.items():
            if hostid == 'success':
                continue

            host_baseline = self.baseline[hostid]
            if host == host_baseline:
                continue

            host_anomalies = {}
            if 'Files' in host:
                baseline_files = host_baseline.get('Files',[])
                anomalous_files = []
                for f in host['Files']:
                    if f not in baseline_files:
                        anomalous_files.append(f)
                if anomalous_files:
                    host_anomalies['Files'] = anomalous_files

            if 'Processes' in host:
                baseline_processes = host_baseline.get('Processes',[])
                anomalous_processes = []
                for p in host['Processes']:
                    if p not in baseline_processes:
                        anomalous_processes.append(p)
                if anomalous_processes:
                    host_anomalies['Processes'] = anomalous_processes

            if host_anomalies:
                anomaly_dict[hostid] = host_anomalies

        return anomaly_dict

    def _process_anomalies(self,anomaly_dict):
        info = deepcopy(self.blue_info)
        for hostid, host_anomalies in anomaly_dict.items():
            assert len(host_anomalies) > 0
            if 'Processes' in host_anomalies:
                connection_type = self._interpret_connections(host_anomalies['Processes'])
                info[hostid][-2] = connection_type
                if connection_type == 'Exploit':
                    info[hostid][-1] = 'User'
                    self.blue_info[hostid][-1] = 'User'
            if 'Files' in host_anomalies:
                malware = [f['Density'] >= 0.9 for f in host_anomalies['Files']]
                if any(malware):
                    info[hostid][-1] = 'Privileged'
                    self.blue_info[hostid][-1] = 'Privileged'

        return info

    def _interpret_connections(self,activity:list):                
        num_connections = len(activity)

        ports = set([item['Connections'][0]['local_port'] \
            for item in activity if 'Connections' in item])
        port_focus = len(ports)

        remote_ports = set([item['Connections'][0].get('remote_port') \
            for item in activity if 'Connections' in item])
        if None in remote_ports:
            remote_ports.remove(None)

        if num_connections >= 3 and port_focus >=3:
            anomaly = 'Scan'
        elif 4444 in remote_ports:
            anomaly = 'Exploit'
        elif num_connections >= 3 and port_focus == 1:
            anomaly = 'Exploit'
        elif 'Service Name' in activity[0]:
            anomaly = 'None'
        else:
            anomaly = 'Scan'

        return anomaly

    # def _malware_analysis(self,obs,hostname):
        # anomaly_dict = {hostname: {'Files': []}}
        # if hostname in obs:
            # if 'Files' in obs[hostname]:
                # files = obs[hostname]['Files']
            # else:
                # return anomaly_dict
        # else:
            # return anomaly_dict

        # for f in files:
            # if f['Density'] >= 0.9:
                # anomaly_dict[hostname]['Files'].append(f)

        # return anomaly_dict


    def _create_blue_table(self, success):
        table = PrettyTable([
            'Subnet',
            'IP Address',
            'Hostname',
            'Activity',
            'Compromised'
            ])
        for hostid in self.info:
            table.add_row(self.info[hostid])
        
        table.sortby = 'Hostname'
        table.success = success
        return table

    def _create_vector(self, success):
        table = self._create_blue_table(success)._rows

        proto_vector = []
        for row in table:
            # Activity
            activity = row[3]
            if activity == 'None':
                value = [0,0,1]
            elif activity == 'Scan':
                value = [0,1,0]
            elif activity == 'Exploit':
                value = [1,0,0]
            else:
                raise ValueError('Table had invalid Access Level')
            proto_vector.extend(value)

            # Compromised
            compromised = row[4]
            if compromised == 'No':
                value = [0,0,0,1]
            elif compromised == 'Unknown':
                value = [0,0,1,0]
            elif compromised == 'User':
                value = [0,1,0,0]
            elif compromised == 'Privileged':
                value = [1,0,0,0]
            else:
                raise ValueError('Table had invalid Access Level')
            proto_vector.extend(value)

        return np.array(proto_vector)

    def get_attr(self,attribute:str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        if agent == 'Blue' and self.output_mode == 'table':
            output = self.get_table()
        else:
            output = self.get_attr('get_observation')(agent)

        return output

    def get_agent_state(self,agent:str):
        return self.get_attr('get_agent_state')(agent)

    def get_action_space(self,agent):
        return self.env.get_action_space(agent)

    def get_last_action(self,agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()


"""
-----------------------------------------------------------------------------
"""

class EnvironmentController:
    """The abstract base controller for all CybORG environment controllers.

    Provides the abstract methods which all CybORG controllers must implement. This includes setup and teardown,
    modifying the state, and pulling out data from the environment.
    When both Simulation and Emulation share common functionality, it is implemented here.


    Attributes
    ----------
    scenario_dict : dict
        the scenario data
    agent_interfaces : dict[str: AgentInterface]
        agent interface object for agents in scenario
    """

    def __init__(self, scenario_path: str, scenario_mod: dict = None, agents: dict = None):
        """Instantiates the Environment Controller.
        Parameters
        ----------
        scenario_path : str
            path to scenario YAML file
        agents : dict, optional
            map from agent name to agent interface of agents to be used in
            environment. If None agents will be loaded from description in
            scenario file (default=None)
        """
        self.hostname_ip_map = None
        self.subnet_cidr_map = None
        # self.scenario_dict = self._parse_scenario(scenario_path, scenario_mod=scenario_mod)
        scenario_dict = self._parse_scenario(scenario_path)
        self.scenario = Scenario(scenario_dict)
        self._create_environment()
        self.agent_interfaces = self._create_agents(agents)
        self.reward = {}
        self.INFO_DICT = {}
        self.action = {}
        self.done = False
        self.observation = {}
        self.INFO_DICT['True'] = {}
        for host in self.scenario.hosts:
            self.INFO_DICT['True'][host] = {'System info': 'All', 'Sessions': 'All', 'Interfaces': 'All', 'User info': 'All',
                                      'Processes': ['All']}
        self.init_state = self._filter_obs(self.get_true_state(self.INFO_DICT['True'])).data
        for agent in self.scenario.agents:
            self.INFO_DICT[agent] = self.scenario.get_agent_info(agent).osint.get('Hosts', {})
            for host in self.INFO_DICT[agent].keys():
                self.INFO_DICT[agent][host]['Sessions'] = agent
        # populate initial observations with OSINT
        for agent_name, agent in self.agent_interfaces.items():
            self.observation[agent_name] = self._filter_obs(self.get_true_state(self.INFO_DICT[agent_name]), agent_name)
            agent.set_init_obs(self.observation[agent_name].data, self.init_state)
            if agent_name == 'Blue':
                self.afterstate = Afterstate(self.observation[agent_name].data)

        self.true_state_pres = []
        self.true_state_after_blues = []
        self.true_state_after_all = []
        self.afterstates = []
        self.scanned_ips = set()
        self.step_counter = 0

    def reset(self, agent: str = None) -> Results:
        """Resets the environment and get initial agent observation and actions.

        Parameters
        ----------
        agent : str, optional
            the agent to get initial observation for, if None will return
            initial white state (default=None)

        Returns
        -------
        Results
            The initial observation and actions of a agent or white team
        """
        self.reward = {}
        self.steps = 0
        self.done = False
        self.init_state = self._filter_obs(self.get_true_state(self.INFO_DICT['True'])).data
        for agent_name, agent_object in self.agent_interfaces.items():
            agent_object.reset()
            self.observation[agent_name] = self._filter_obs(self.get_true_state(self.INFO_DICT[agent_name]), agent_name)
            agent_object.set_init_obs(self.observation[agent_name].data, self.init_state)
            if agent_name == 'Blue':
                self.afterstate = Afterstate(self.observation[agent_name].data)

        # self.true_state_pres = []
        # self.true_state_after_blues = []
        # self.true_state_after_all = []
        # self.scanned_ips = set()
        # self.step_counter = 0

        if agent is None:
            return Results(observation=self.init_state)
        else:
            return Results(observation=self.observation[agent].data,
                           action_space=self.agent_interfaces[agent].action_space.get_action_space())
        

    def _update_scanned(self):
        if self.step_counter <= 0:
            return

        action = self.get_last_action(agent='Red')
        if action.__class__.__name__ == 'DiscoverNetworkServices':
            red_obs = deepcopy(self.get_last_observation(agent='Red').data)
            success = red_obs['success']
            if success:
                ip = red_obs.popitem()[0]
                self.scanned_ips.add(ip)

    def make_true_state_vector(self):
        state = deepcopy(self._filter_obs(self.get_true_state(self.INFO_DICT['True'])).data)
        self._update_scanned()
        # table = PrettyTable([
        #     'Subnet',
        #     'IP Address',
        #     'Hostname',
        #     'Known',
        #     'Scanned',
        #     'Access',
        #     ])
        # output = {"known":[],"scanned":[],"access":[]} # ,"persistent":[] red agent is always user0!
        output_list = []
        for hostid in state:
            # print(hostid,flush=True)
            host = state[hostid]
            if isinstance(host, dict) and "Interface" in host:
                for interface in host['Interface']:
                    ip = interface['IP Address']
                    if str(ip) == '127.0.0.1':
                        continue
                    if 'Subnet' not in interface:
                        continue
                    # subnet = interface['Subnet']
                    # hostname = host['System info']['Hostname']
                    action_space = self.get_action_space(agent = 'Red')
                    known = action_space['ip_address'][ip]
                    scanned = True if str(ip) in self.scanned_ips else False
                    if scanned:
                        assert known, "assumpution failed that there are 3 possible discovery states unknown->known->scanned"

                    access = self._determine_red_access(host['Sessions'])

                    output_list.extend([not (known or scanned), known and not scanned, scanned]+access)
                    # output["known"].append(action_space['ip_address'][ip])
                    # output["scanned"].append(True if str(ip) in self.scanned_ips else False)
                    # output["access"].append(self._determine_red_access(host['Sessions']))


                # table.add_row([subnet,str(ip),hostname,known,scanned,access])
        
        # table.sortby = 'Hostname'
        # table.success = success
        return np.array(output_list)

    def _determine_red_access(self,session_list):
        for session in session_list:
            if session['Agent'] != 'Red':
                continue
            privileged = session['Username'] in {'root','SYSTEM'}
            return [0,0,1] if privileged else [0,1,0]#'Privileged' if privileged else 'User'

        return [1,0,0]# 'None'

    def record_additional_states(self, pre, after_blue, after_all, afterstate):
        if len(self.true_state_after_all)>0:
            last_red = self.true_state_after_all[len(self.true_state_after_all)-1]
            assert last_red.shape == (36,)
            assert pre.shape == (36,)
            assert np.all(last_red == pre), "broken assumption that states after red action and before blue action are always equal"
            # if not np.all(self.true_state_after_all[len(self.true_state_after_all)-1] == pre):
            #     print(f"NOT EQUAL!:  {self.true_state_after_all[len(self.true_state_after_all)-1] == pre}")
        self.true_state_pres.append(pre)
        self.true_state_after_blues.append(after_blue)
        self.true_state_after_all.append(after_all)
        self.afterstates.append(afterstate)
        

    def pop_additional_states_sequences(self):
        return_tuple = (self.true_state_pres, self.true_state_after_blues, self.true_state_after_all, self.afterstates)

        self.true_state_pres = []
        self.true_state_after_blues = []
        self.true_state_after_all = []
        self.afterstates = []
        self.scanned_ips = set()
        self.step_counter = 0

        return return_tuple

    def step(self, agent: str = None, action: Action = None, skip_valid_action_check: bool = False) -> Results:
        """Perform a step in the environment for given agent.

        Parameters
        ----------
        agent : str, optional
            the agent to perform step for (default=None)
        action : Action/
            the action to perform

        Returns
        -------
        Results
            the result of agent performing the action
        """
        true_obs_pre_any_actions = self.make_true_state_vector()

        # for each agent:
        next_observation = {}
        # all agents act on the state
        for i, (agent_name, agent_object) in enumerate(self.agent_interfaces.items()):
            # pass observation to agent to get actio

            if agent is None or action is None or agent != agent_name:
                agent_action = agent_object.get_action(self.observation[agent_name])

            else:
                agent_action = action
            if not self.test_valid_action(agent_action, agent_object) and not skip_valid_action_check:

                agent_action = InvalidAction(agent_action)
            self.action[agent_name] = agent_action

            # perform action on state
            next_observation[agent_name] = self._filter_obs(self.execute_action(self.action[agent_name]), agent_name)
            if agent_name == "Blue":
                assert i == 0, "Assert that blue's is the first action failed, this means the saved true state will include the effects of the red action"
                # Assumes blue is first agent in the list as it is added first in _create_agents() since it is the first agent in Scenario2.yaml
                true_obs_post_blue_action = self.make_true_state_vector()

                #Get afterstate
                agent_session = list(self.get_action_space(agent_name)['session'].keys())[0]
                agent_observation = self._filter_obs(
                    self.execute_action(Monitor(session=agent_session, agent='Blue')), agent_name)
                first_action_success = self.observation[agent_name].success
                self.observation[agent_name].combine_obs(agent_observation)
                self.observation[agent_name].set_success(first_action_success)
                agent_object.update(self.observation[agent_name])
                self.afterstate.last_action = self.get_last_action(agent_name)
                afterstate_vector = self.afterstate.observation_change(self.observation[agent].data)
                #print('After Action: ', afterstate_vector)

        # get true observation
        true_observation = self._filter_obs(self.get_true_state(self.INFO_DICT['True'])).data

        # Blue update step.
        # New idea: run the MONITOR action for the Blue agent, and update the observation.

        # pass training information to agents
        for agent_name, agent_object in self.agent_interfaces.items():

            # determine done signal for agent
            done = self.determine_done(next_observation, true_observation, self.action[agent_name])
            self.done = done or self.done
            # determine reward for agent
            reward = agent_object.determine_reward(next_observation, true_observation,
                                                   self.action, self.done)
            self.reward[agent_name] = reward + self.action[agent_name].cost
            if agent_name != agent:
                # train agent using obs, reward, previous observation, and done
                agent_object.train(Results(observation=self.observation[agent_name].data, reward=reward,
                                           next_observation=next_observation[agent_name].data, done=self.done))
            self.observation[agent_name] = next_observation[agent_name]

            agent_object.update(self.observation[agent_name])

            # if self.verbose and type(self.action[agent_name]) != Sleep and self.observation[agent_name].dict['success'] == True:
            #    print(f"Step: {self.steps}, {agent_name}'s Action Choice: {type(self.action[agent_name]).__name__}, "
            #          f"Reward: {reward}")

        # Update Blue's observation with the latest information before returning.
        for agent_name, agent_object in self.agent_interfaces.items():
            if agent_name == 'Blue':
                agent_session = list(self.get_action_space(agent_name)['session'].keys())[0]
                agent_observation = self._filter_obs(
                    self.execute_action(Monitor(session=agent_session, agent='Blue')), agent_name)
                first_action_success = self.observation[agent_name].success
                self.observation[agent_name].combine_obs(agent_observation)
                self.observation[agent_name].set_success(first_action_success)
                agent_object.update(self.observation[agent_name])

                #self.afterstate.last_action = self.get_last_action(agent_name)
                #vec2 = self.afterstate.observation_change(self.observation[agent].data)
                #print('Latest Info: ', vec2)
        # if done then complete other agent's turn

        #print('Delta: ', afterstate_vector - vec2)

        true_obs_post_all_actions = self.make_true_state_vector()

        self.record_additional_states(true_obs_pre_any_actions, true_obs_post_blue_action, true_obs_post_all_actions, afterstate_vector)

        if agent is None:
            result = Results(observation=true_observation, done=self.done)
        else:
            result = Results(observation=self.observation[agent].data, done=self.done, reward=round(self.reward[agent], 1),
                             action_space=self.agent_interfaces[agent].action_space.get_action_space(),
                             action=self.action[agent])
        
        self.step_counter += 1
        
        return result

    def execute_action(self, action: Action) -> Observation:
        """Execute an action in the environment"""
        raise NotImplementedError

    def determine_done(self,
                       agent_obs: dict,
                       true_obs: dict,
                       action: Action) -> bool:
        """Determine if environment scenario goal has been reached.

        Parameters
        ----------
        agent_obs : dict
            the agents last observation
        true_obs : dict
            the current white state
        action : Action
            the agents last action performed

        Returns
        -------
        bool
            whether goal was reached or not
        """
        return False

    def start(self, steps: int = None, log_file=None):
        """Start the environment and run for a specified number of steps.

        Parameters
        ----------
        steps : int
            the number of steps to run for
        log_file : File, optional
            a file to write results to (default=None)

        Returns
        -------
        bool
            whether goal was reached or not
        """
        done = False
        max_steps = 0
        if steps is None:
            while not done:
                max_steps += 1
                _, _, done = self.step()
            print('Red Wins!')  # Junk Test Code
        else:
            for step in range(steps):
                max_steps += 1
                results = self.step()
                done = results.done
                if step == 500:
                    print(step)  # Junk Test Code
                if done:
                    print(f'Red Wins at step {step}')  # Junk Test Code
                    break
        for agent_name, agent in self.agent_interfaces.items():
            agent.end_episode()
            # print(f"{agent_name}'s Reward: {self.reward[agent_name]}")
        if log_file is not None:
            log_file.write(
                f"{max_steps},{self.reward['Red']},{self.reward['Blue']},"
                f"{self.agent_interfaces['Red'].agent.epsilon},"
                f"{self.agent_interfaces['Red'].agent.gamma}\n"
            )
        return done

    def get_true_state(self, info: dict) -> Observation:
        """Get current True state

        Returns
        -------
        Observation
            current true state
        """
        raise NotImplementedError

    def get_agent_state(self, agent_name: str) -> Observation:
        return self.get_true_state(self.INFO_DICT[agent_name])

    def get_last_observation(self, agent: str) -> Observation:
        """Get the last observation for an agent

        Parameters
        ----------
        agent : str
            name of agent to get observation for

        Returns
        -------
        Observation
            agents last observation
        """
        return self.observation[agent]

    def get_action_space(self, agent: str) -> dict:
        """
        Gets the action space for a chosen agent
        agent: str
            agent selected
        """
        if agent in self.agent_interfaces:
            return self.agent_interfaces[agent].action_space.get_action_space()
        raise ValueError(f'Agent {agent} not in agent list {self.agent_interfaces.values()}')

    def get_observation_space(self, agent: str) -> dict:
        """
                Gets the observation space for a chosen agent
                agent: str
                    agent selected
                """
        if agent in self.agent_interfaces:
            return self.agent_interfaces[agent].get_observation_space()
        raise ValueError(f'Agent {agent} not in agent list {self.agent_interfaces.values()}')

    def get_last_action(self, agent: str) -> Action:
        """
                Gets the observation space for a chosen agent
                agent: str
                    agent selected
                """
        return self.action[agent] if agent in self.action else None



    def restore(self, filepath: str):
        """Restores the environment from file

        Parameters
        ----------
        filepath : str
            path to file to restore env from
        """
        raise NotImplementedError

    def save(self, filepath: str):
        """Saves the environment to file

        Parameters
        ----------
        filepath : str
            path to file to save env to
        """
        raise NotImplementedError

    def pause(self):
        """Pauses the environment"""
        pass

    def shutdown(self, teardown: bool = True) -> bool:
        """Shutdown environment, deleting/terminating resources
        as required

        Parameters
        ----------
        teardown : bool, optional
            if True environment resources will be terminated if applicable,
            otherwise resources will not be terminated (allowing them to be
            reused if desired) (default=True)

        Returns
        -------
        bool
            True if the environment was shutdown without issue
        """
        raise NotImplementedError

    def _parse_scenario(self, scenario_file_path: str, scenario_mod: dict = None):
        with open(scenario_file_path) as fIn:
            scenario_dict = yaml.load(fIn, Loader=yaml.FullLoader)
        return scenario_dict

    def _create_agents(self, agent_classes: dict = None) -> dict:
        agents = {}

        for agent_name in self.scenario.agents:
            agent_info = self.scenario.get_agent_info(agent_name)
            if agent_classes is not None and agent_name in agent_classes:
                agent_class = agent_classes[agent_name]
            else:
                agent_class = getattr(sys.modules['CybORG.Agents'],
                                      agent_info.agent_type)
            agents[agent_name] = AgentInterface(
                agent_class,
                agent_name,
                agent_info.actions,
                agent_info.reward_calculator_type,
                allowed_subnets=agent_info.allowed_subnets,
                wrappers=agent_info.wrappers,
                scenario=self.scenario
            )
        return agents

    def _create_environment(self):
        raise NotImplementedError

    def _filter_obs(self, obs: Observation, agent_name=None):
        """Filter obs to contain only hosts/subnets in scenario network """
        if agent_name is not None:
            subnets = [self.subnet_cidr_map[s] for s in self.scenario.get_agent_info(agent_name).allowed_subnets]
        else:
            subnets = list(self.subnet_cidr_map.values())

        obs.filter_addresses(
            ips=self.hostname_ip_map.values(),
            cidrs=subnets,
            include_localhost=False
        )
        return obs

    def test_valid_action(self, action: Action, agent: AgentInterface):
        # returns true if the parameters in the action are in and true in the action set else return false
        action_space = agent.action_space.get_action_space()
        # first check that the action class is allowed
        if type(action) not in action_space['action'] or not action_space['action'][type(action)]:
            return False
        # next for each parameter in the action
        for parameter_name, parameter_value in action.get_params().items():
            if parameter_name not in action_space:
                continue
            if parameter_value not in action_space[parameter_name] or not action_space[parameter_name][parameter_value]:
                return False
        return True

    def get_reward_breakdown(self, agent:str):
        return self.agent_interfaces[agent].reward_calculator.host_scores


        




