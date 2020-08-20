import setup_path
import airsim

from ActorCritic import *
from utils import *
from vehicles import Car, Drone


def play_game(logger, uuid, pos=(0, 0, -1), goal=(120, 35), uav_size=(0.29*3, 0.98*2), hfov = radians(90), coll_thres=5, yaw=0,
              limit_yaw=5, step=0.1):

    # -- create episode and vehicle objects --------
    episode = Episode(uuid, logger)

    drone = Drone()
    car = Car()

    # --- Plan path to goal -------------------------------------------------------------------------------------
    topview, filename = get_topview_image(190, drone=drone)
    obstacles = PathPlanningObstacles(filename, proportion_x=1.6, proportion_y=1.6)
    goals, planner = path_planning(obstacles=obstacles, topview=topview, x_goal=goal, x_init=pos[:2])

    # --- send vehicle and drone to initial positions (random at each game/episode) ------------------------------
    drone.move(pos, yaw)
    car.move(pos, yaw)

    # ------------------------------------------------------------------------------------------------------------
    env = Environment(init_pose=pos, drone=drone, car=car, planner=planner, goals=goals)

    state = env.reset(pos, yaw)
    print('Init state:', state)

    # --- Load models -------------------------------------------------------------------------------------------

    trainer = Trainer(env)
    a2c, optimizer = trainer.load_a2c()

    # --- Start Game ---------------------------------------------------------------------------------------------
    total_entropy = 0
    limit = 30
    count_down = 10
    done = False

    while (len(env.goals) or count_down) and env.dt < limit:

        value, policy_dist = a2c.forward(state)   # each a vector of size n_actions (i.e. 10 for now)

        value = value.detach().numpy()[0, 0]
        distribution = policy_dist.detach().numpy()

        action = np.random.choice(env.action_space.n, p=np.squeeze(distribution))  # weighted choosing based on output
        log_prob = torch.log(policy_dist.squeeze(0)[action])
        entropy = -np.sum(np.mean(distribution) * np.log(distribution))
        total_entropy +=entropy

        print('Action taken:', action)
        new_state, reward, done, _ = env.step(env.action_space.action(action), delay=1)

        episode.add_experience(Experience(state, action, reward, new_state, value, log_prob, episode.uuid))

        if done:
            print('Target reached.')
            episode.logger.info('target_reached')
            count_down -= 1
            if not count_down:
                break

        state = new_state

    # --- Game ended, running training cycle  -------------------------------------------------------------------

    if done:
        print('Drone Reached Final Target')
        episode.logger.info('final_target_reached')

    print('Running training phase')
    trainer.train(episode, state, total_entropy)
    trainer.save_model()

    # drone.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)


# ---- MAIN ---------------------------------------------------------

if __name__ == '__main__':
    episode_uuid = round(time.time())
    logger = setup_logger('logger', 'episodes', f'episode_{episode_uuid}_{time.strftime("%a,%d_%b_%Y_%H_%M_%S")}.txt')
    logger.info(f'Episode: {episode_uuid} - START -')

    play_game(logger, episode_uuid)
    logger.info('GAME_END')
