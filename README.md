# Yahtzee Reinforcement Learning Iterations
This repo is a collection of the environments, agents, and algorithms I've developed along the way towards reaching the current best state, which is V9. 
All of the old files are in folders for storage purposes - they are not runnable unless they are all brought together in the same directory and imported to one another properly (not that you would want to run any of them, they mostly stink).
I decided in favor of rapid iteration and not getting bogged down in organization and infrastructure. Now that the agent is in a somewhat final state for what I want to achieve, I will improve documentation both in the code and on a dedicated docs page. 
Next steps involve running a more rigorous analysis of the agent's performance and then collating results into a more presentable paper. 

It took a lot of iterations, but the current agent could very well play against a human and win a decent amount of games. The key ended up being striking the right balance of condensing the state space enough to be explorable but not too much so as to lose detail for the agent to learn from. This, combined with a good incentive structure in the reward function resulted in the best performance I've seen through all the iterations.
