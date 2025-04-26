from environment import PlaneNavigationEnv


def main():
    # initalize environment and get initale state
    env = PlaneNavigationEnv()
    observation, info = env.reset()

    for _ in range(100):
        env.render()
        print(env.action_space)
        action = env.action_space.sample()  # random action for now
        observation, reward, terminated, truncated, info = env.step(action)

        print(observation, reward, terminated, truncated, info)
        
        # if destination is reached or scenario is ended
        if terminated or truncated:
            print(f"Episode finished! Reward: {reward}")
            break

    env.close()


if __name__ == '__main__':
    main()