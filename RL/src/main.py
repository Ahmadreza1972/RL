from processore import QLearningAgentWithGraph
from writer import Writer
from visualization import Visualization

def main():
    agent = QLearningAgentWithGraph()
    write= Writer()
    viso=Visualization()
    
    #agent.train()
    
    write.save_q_table(agent._q_table)
      
    viso.show_qtable()


if __name__ == "__main__":
    main()

    
    
