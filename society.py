'''
create a class for agent where each agent has a a system prompt, a memory, any tools, and an output function.
then we need a class for society where we need function to communicate.
    - we should input a graph and a way that determines what one pass of communication looks like
    for eg. in fully connected graph, one pass is everyone talking once based on other's outputs
    but in heirarchical setup, one pass is all child agents talking once to the parent. 
'''

from llm import LLM
from vLLM import vLLM_call 

# instantiate the vLLM class
# vLLM = vLLM_call("FreedomIntelligence/HuatuoGPT-o1-7B")
vLLM = vLLM_call("facebook/opt-125m")


def extract_opention(answer):
    start = answer.find("<Opinion>")
    end = answer.find("</Opinion>")
    return answer[start+9:end]

class Agent:
    def __init__(self, name, id, system_prompt, memory=None, tools=None, model_name="gemma_base"):
        """
        Initialize an agent with a system prompt, memory, tools, and an output function.

        :param name: str, unique identifier for the agent
        :param system_prompt: str, a guiding prompt for the agent
        :param memory: list, stores agent's memory (default: empty list)
        :param tools: list, tools available to the agent (default: None)
        """
        self.name = name
        self.id = id
        self.system_prompt = system_prompt
        self.memory = memory or ""
        self.tools = tools or []
        self.model_name = model_name
        self.last_utterance = ""

    def return_agent_inits(self):
        return self.name, self.id, self.system_prompt, self.memory, self.tools
    
    def update_memory(self, update):
        self.memory+=update

    def process_input_parallel_string(self, inputs, background=None):
        message = f"role: system, content: {self.system_prompt}"
            
        if(len(self.tools)>1):
            tool_output = self.tool_output(inputs, background)
            message += "role: user, content: Memory: {self.memory}\n{self.tools[0]}: {tool_output}\nCurrent Input: {inputs} \n Answer: Let's think step by step."
        else: 
            message += f"role: user, content: Question: {inputs} \n {self.memory}\n Answer: Let's think step by step."

        return message

    def update_memory_parallel(self, response):
        self.memory+="Your Opinion:" + response + "\n\n\n"
        self.last_utterance = response
    
    def process_input(self, inputs, background=None):
        """
        Process input using the agent's system prompt, memory, and tools.

        :param inputs: str, input to the agent
        :return: str, agent's output
        """
        # Construct the messages array
        messages = [
            {"role": "system", "content": self.system_prompt},
            ]
        if(len(self.tools)>1):
            tool_output = self.tool_output(inputs, background)
            messages.append({"role": "user", "content": f"Memory: {self.memory}\n{self.tools[0]}: {tool_output}\nCurrent Input: {inputs} \n Answer: Let's think step by step."})
        else: 
            messages.append({"role": "user", "content": f"Question: {inputs} \n {self.memory}\n Answer: Let's think step by step."})

        # Generate response using LLM()
        response = LLM(messages, self.model_name)

        # extract the <Opinion> part
        # response = extract_opention(response)

        # Update memory with the new response
        self.memory+="Your Opinion:" + response + "\n\n\n"
        self.last_utterance = response

        return response
    
    def tool_output(self, inputs, background):
        if("background_knowledge" in self.tools):
            return background
    
class Society:
    def __init__(self, agents, graph):
        """
        Initialize the society with agents, a communication graph, and a strategy.

        :param agents: list of Agent, the agents in the society
        <Assumption: the list of agents is in order that the uder wants them to communicate in>
        :param graph: dict, adjacency list representation of communication graph
        :param communication_strategy: callable, defines one pass of communication
        """
        self.agents = agents
        self.graph = graph 
    
    def update_neighbour_memory_parallel(self, agent, utterance, background):
        update_child_memory = f"{agent.name}'s Opinion: {utterance} \n\n\n"
        for child in self.graph[agent.name]:
            child.update_memory(update_child_memory)
            # print(f"{agent.name} communicated with {child.name}")
            # print(f"Memory of {child.name}: {child.memory}")


    # how to determine what is a round of conversation? Depends on the graph that you input
    # decomposition? Maybe fornow take tasks that can be decomposed easily?
    def communicate_one_round(self, input, background):
        communication_graph = self.graph
        set_of_agents = self.agents
        for agent in set_of_agents:
            # breakpoint()
            # print("Currently processing agent: ", agent.name)
            agent_output = agent.process_input(input, background)
            update_child_memory = f"{agent.name}'s Opinion: {agent_output} \n\n\n"
            for child in communication_graph[agent.name]:
                child.update_memory(update_child_memory)
                # print(f"{agent.name} communicated with {child.name}")
                # print(f"Memory of {child.name}: {child.memory}")
        
        # for agent in set_of_agents:
        #     print("*****************")
        #     print(f"Memory of agent {agent.name}: {agent.memory}")

    def run_simulation(self, num_rounds, input, background):
        for i in range(num_rounds):
            # print(f"Round {i+1}")
            self.communicate_one_round(input, background)
            # print("*****************")
            

class ListOfSocities:
    def __init__(self, list_of_socities, agents, graph):
        self.list_of_socities = list_of_socities
        self.agents = agents
        self.graph = graph
    
    def run_simulation_parallel(self, num_rounds, questions, background):
        
        set_of_agents = self.agents
        for i in range(num_rounds):
            for agent in set_of_agents:
                llm_input_list = []
                for society, question in zip(self.list_of_socities, questions):
                    llm_input_list.append(society.agents[agent.id].process_input_parallel_string(question, background))
                
                responses = vLLM.call(llm_input_list)
                # print(responses[0])
                for society, response in zip(self.list_of_socities, responses):
                    # breakpoint()
                    society.agents[agent.id].update_memory_parallel(response)
                    # print(f"Memory of agent {society.agents[agent.id].name}: {society.agents[agent.id].memory}")
                    society.update_neighbour_memory_parallel(society.agents[agent.id], response, background)

                    



def create_random_graph(agents, rule_fn, percentage_connectedness):
    # find the number of edges based on percentage connectedness
    num_edges = int(percentage_connectedness*len(agents)*(len(agents)-1)/2)

    # first create a spanning tree that adds edges according to rule_fn over the agents  
    # then add extra edges to reach the desired percentage connectedness           
    graph = {}
    




     

        






