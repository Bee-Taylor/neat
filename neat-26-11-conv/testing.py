import gym
BIPEDAL_WALKER = gym.make("BipedalWalker-v3")
BIPEDAL_WALKER_HARD = gym.make("BipedalWalkerHardcore-v3")

def bipedal_walker(network, render, hard, distance):
    global BIPEDAL_WALKER_HARD
    global BIPEDAL_WALKER
    if hard:
        env = BIPEDAL_WALKER_HARD
    else:
        env = BIPEDAL_WALKER
    reward_sum = 0
    observation = env.reset()
    done = False
    ratio = [0.0, 0.0]
    while not done:
        outputs = network.run(observation)
        outputs.reverse()
        for output in range(len(outputs)):
            if outputs[output] < -1:
                outputs[output] = -1
            elif outputs[output] > 1:
                outputs[output] = 1
            else:
                ratio[0] += 1
            ratio[1] += 1
        observation, reward, done, _ = env.step(outputs)
        if distance:
            reward_sum += observation[2]
        else:
            reward_sum += reward
        if render:
            env.render()
    #print(ratio[0] / ratio[1])
    return reward_sum

def bipedal_walker_reward(network, render, hard, distance, counter = 0):
    BIPEDAL_WALKER = gym.make("BipedalWalker-v3")
    BIPEDAL_WALKER_HARD = gym.make("BipedalWalkerHardcore-v3")
    if hard:
        env = BIPEDAL_WALKER_HARD
    else:
        env = BIPEDAL_WALKER
    reward_sum = 0
    observation = env.reset()
    done = False
    ratio = [0.0, 0.0]
    reward_sum_tmp = 0
    loop_counter = 0
    while not done:
        outputs = network.run(observation)
        outputs.reverse()
        for output in range(len(outputs)):
            if outputs[output] < -1:
                outputs[output] = -1
            elif outputs[output] > 1:
                outputs[output] = 1
            else:
                ratio[0] += 1
            ratio[1] += 1
        observation, reward, done, _ = env.step(outputs)
        if distance:
            reward_sum += observation[2]
        else:
            reward_sum += reward
        reward_sum_tmp += reward
        if loop_counter >= 9:
            loop_counter = 0
            if reward_sum_tmp > 0:
                network.keep_changes()
                #network.revert_changes()
            else: 
                network.revert_changes()
        if render:
            env.render()
        loop_counter += 1

    #print(ratio[0] / ratio[1])
    network.revert_changes()
    if counter < 10 and not render:
        return reward_sum + bipedal_walker_reward(network, render,hard, distance, counter = counter + 1)
    else:
        return reward_sum


def bipedal_walker_no_render_easy_distance(network):
    return bipedal_walker(network, False, False, True)

def bipedal_walker_no_render_easy_in_built(network):
    return bipedal_walker(network, False, False, False)

def bipedal_walker_no_render_hard_distance(network):
    return bipedal_walker(network, False, True, True)

def bipedal_walker_no_render_hard_in_built(network):
    return bipedal_walker(network, False, True, False)

def bipedal_walker_no_render_easy_distance_reward(network):
    return sum(bipedal_walker_reward(network, False, False, True) for i in range(10))

def bipedal_walker_no_render_easy_in_built_reward(network):
    return sum(bipedal_walker_reward(network, False, False, False) for i in range(10))

def bipedal_walker_no_render_hard_distance_reward(network):
    return bipedal_walker_reward(network, False, True, True)

def bipedal_walker_no_render_hard_in_built_reward(network):
    return bipedal_walker_reward(network, False, True, False)


def walker(conn):
    keep_going = True
    first = True
    while keep_going:
        if not conn.empty() or first:
            keep_going, network, choice = conn.get()
        first =  False
        hard = False #bool(choice%2)
        distance = bool(choice/2%1)
        bipedal_walker(network, True, hard, distance)
