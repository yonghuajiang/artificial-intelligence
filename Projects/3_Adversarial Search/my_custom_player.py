
from sample_players import DataPlayer
import random
from isolation.isolation import _WIDTH, _HEIGHT

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only *required* method. You can modify
    the interface for get_action by adding named parameters with default
    values, but the function MUST remain compatible with the default
    interface.

    **********************************************************************
    NOTES:
    - You should **ONLY** call methods defined on your agent class during
      search; do **NOT** add or call functions outside the player class.
      The isolation library wraps each method of this class to interrupt
      search when the time limit expires, but the wrapper only affects
      methods defined on this class.

    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    **********************************************************************
    """
    def get_action(self, state):

        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        if state.ply_count < 2: self.queue.put(random.choice(state.actions()))
        for depth in range(3,100):
            best_action = self.alpha_beta_search(state, depth) 
            self.queue.put(best_action)

    def alpha_beta_search(self, state, depth):

        def min_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: 
                value = self.score(state)
                return value
            value = float("inf")
            action_score = [(action,self.score(state.result(action))) for action in state.actions()]
            action_score.sort(key=lambda tup: tup[1],reverse=True)
            action_list = [action[0] for action in action_score]
            for action in state.actions():
                value = min(value, max_value(state.result(action), alpha, beta, depth - 1))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value

        def max_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            action_score = [(action,self.score(state.result(action))) for action in state.actions()]
            action_score.sort(key=lambda tup: tup[1])
            action_list = [action[0] for action in action_score]
            for action in action_list:
                value = max(value, min_value(state.result(action), alpha, beta, depth - 1))
                if value >= beta:
                    return value
                alpha = max(alpha,value)
            return value
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for action in state.actions():
            v = min_value(state.result(action), alpha, beta, depth - 1)
            alpha = max(alpha,v)
            if v > best_score:
                best_score = v
                best_move = action
        return best_move

  
    def score(self, state):
        sim_score = 0
        for _ in range(1000):
            sim_score += self.mc_simulation(state)
        return sim_score/1000    
        
    def mc_simulation(self,state):
        if state.terminal_test(): 
            if state.utility(self.player_id) == float('inf'): 
                return 1
            else: return 0
        else: 
            legal_action = state.actions()
            action = random.choice(legal_action)
            new_state = state.result(action)
            return self.mc_simulation(new_state)
        
    
           
        