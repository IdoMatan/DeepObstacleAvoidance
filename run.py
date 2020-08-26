import setup_path
import airsim

from ActorCritic import *
from utils import *
from vehicles import Car, Drone


def play_game(logger, uuid, pos=(0, 0, -1), goal=(120, 35), uav_size=(0.29*3, 0.98*2), hfov=radians(90), coll_thres=5, yaw=0,
              limit_yaw=5, step=0.1):

    # -- create episode and vehicle objects --------
    episode = Episode(uuid, logger)

    lead_drone = Drone(name='Drone1', mode='leader', uav_size=uav_size)
    follower_drone = Drone(name='Drone2', mode='follower')

    # --- Plan path to goal -------------------------------------------------------------------------------------
    topview, filename = get_topview_image(190, drone=lead_drone)
    obstacles = PathPlanningObstacles(filename, proportion_x=1.6, proportion_y=1.6)
    goals, planner = path_planning(obstacles=obstacles, topview=topview, x_goal=goal, x_init=pos[:2])

    # --- send vehicle and drone to initial positions (random at each game/episode) ------------------------------
    lead_drone.move(pos, yaw)
    follower_drone.move(pos, yaw)

    # --- Load models -------------------------------------------------------------------------------------------

    # trainer = Trainer(env)
    # a2c, optimizer = trainer.load_a2c()

    # --- Start Game ---------------------------------------------------------------------------------------------
    limit = 1200
    dt = 0
    goal = goals.pop(0)
    lead_drone.predictControl = AvoidLeft(hfov, coll_thres, yaw, limit_yaw, step)
    follower_drone.leader = lead_drone
    pos = list(pos)
    while len(goals) and dt < limit:
            # get response
            pos, yaw, target_dist = lead_drone.step(goal, pos)

            follower_drone.follow(pos, yaw)
            follower_drone.save_leading_pic()
            dt += 1

            if target_dist < 1:
                print('Target reached.')
                if not len(goals):
                    break
                goal = goals.pop(0)
                dt = 0

    # --- Game ended -----------------------------------------------------------------------------------


# ---- MAIN ---------------------------------------------------------

if __name__ == '__main__':
    episode_uuid = round(time.time())
    logger = setup_logger('logger', 'episodes', f'episode_{episode_uuid}_{time.strftime("%a,%d_%b_%Y_%H_%M_%S")}.txt')
    logger.info(f'Episode: {episode_uuid} - START -')

    play_game(logger, episode_uuid)
    logger.info('GAME_END')
