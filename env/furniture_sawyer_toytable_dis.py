""" Disassemble version of FurnitureSawyerToyTable """
import numpy as np

import env.transform_utils as T
from env.furniture_sawyer_toytable import FurnitureSawyerToyTableEnv
from env.models import furniture_name2id
from util import clamp
from util.logger import logger


class FurnitureSawyerToyTableDisEnv(FurnitureSawyerToyTableEnv):
    """
    Diassemble table leg and top using Sawyer
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        config.preassembled = [0]
        super().__init__(config)
    
    def _try_connect(self, part1=None, part2=None):
        """
        Disconnects all parts attached to part1
        part1, part2 are names of the body
        """
        assert part1 is not None and part2 is None
        for i, (id1, id2) in enumerate(
            zip(self.sim.model.eq_obj1id, self.sim.model.eq_obj2id)
        ):
            p1 = self.sim.model.body_id2name(id1)
            p2 = self.sim.model.body_id2name(id2)
            if part1 in [p1, p2]:
                # setup eq_data
                # self.sim.model.eq_data[i] = T.rel_pose(
                #     self._get_qpos(p1), self._get_qpos(p2)
                # )
                self.sim.model.eq_active[i] = 0

    def _compute_reward(self, action):
        return 0, False, {}


def main():
    from config import create_parser

    parser = create_parser(env="FurnitureSawyerToyTableEnv")
    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Sawyer environment
    env = FurnitureSawyerToyTableDisEnv(config)
    env.run_manual(config)

    # import pickle
    # with open("demos/Sawyer_toy_table_0022.pkl", "rb") as f:
    #     demo = pickle.load(f)
    # env.reset()
    # print(len(demo['actions']))

    # from util.video_recorder import VideoRecorder
    # vr = VideoRecorder()
    # vr.add(env.render('rgb_array')[0])
    # for ac in demo['actions']:
    #     env.step(ac)
    #     vr.add(env.render('rgb_array')[0])
    # vr.save_video('test.mp4')


if __name__ == "__main__":
    main()
