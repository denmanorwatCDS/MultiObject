from gym.envs.robotics import robot_env, rotations, robot_env
from gym import utils, spaces
from gym.envs.robotics import utils as gym_robotics_utils
import numpy as np
import os
import xml.etree.ElementTree as ET
from gym import error
import copy
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class MultipleFetchPickAndPlaceEnv(robot_env.RobotEnv, utils.EzPickle):
    def __init__(self, seed = None, reward_type = 'sparse', object_qty = 4, with_repeat = True, 
                 object_names = ['ball', 'box', 'desk', 'hammer']):
        
        self.colors = ['1 0 0 1', '0 1 0 1', '0 0 1 1', '1 1 0 1', '0 1 1 1', '1 0 1 1']
        self.tints = ['0.25 0 0 1', '0 0.25 0 1', '0 0 0.25 1', '0.25 0.25 0 1', '0 0.25 0.25 1', '0.25 0 0.25 1']
        
        self.gripper_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0
        }
        self.n_substeps = 20
        self.n_actions = 4
        self.gripper_extra_height = 0.2
        self.block_gripper = False
        self.has_object = True
        self.distance_threshold = 0.1
        for object_name in object_names:
            assert object_name in ['ball', 'box', 'desk', 'hammer'],\
            'Supported items are: ball, box, desk, hammer.{} is not supported'.format(object_name)
        self.object_names = object_names
        
        self.object_qty = object_qty
        self.with_repeat = with_repeat
        self.seed(seed)
        self.created_object_names, self.goal = self._initialize_sim()
        obs = self._get_obs()
        
        self.initial_state = copy.deepcopy(self.sim.get_state())

        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(1, obs['achieved_goal'].shape[1]), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def _create_multiobject_xml(self):
        prefix_to_folder = os.path.join(os.path.dirname(__file__), 'simulation_data', 'xmls')
        model_path = os.path.join(prefix_to_folder, 'pick_and_place.xml')
        xml_tree = ET.parse(model_path)
        root = xml_tree.getroot()
        worldbody_idx = None

        for i, child in enumerate(root):
            if 'worldbody' == child.tag:
                worldbody_idx = i
                break
        
        worldbody = root[worldbody_idx]
        
        sampled_colors_idx = self.np_random.choice(len(self.colors), self.object_qty, replace=False)
        sampled_colors = [self.colors[i] for i in sampled_colors_idx]
        sampled_tints = [self.tints[i] for i in sampled_colors_idx]
        sampled_object_names = self.np_random.choice(self.object_names, self.object_qty, replace = self.with_repeat)
        created_object_names = []
        object_postfixes = {'{}'.format(object_name): 0 for object_name in sampled_object_names}
        
        for object_name, object_color, object_tint in zip(sampled_object_names, sampled_colors, sampled_tints):
            # Modify names for conflict resolution in case two same objects present in one scene
            # And change colors of geoms
            cur_object = ET.parse(os.path.join(prefix_to_folder, '{}.xml'.format(object_name))).getroot()
            old_cur_object_name = cur_object.attrib['name']
            new_cur_object_name = old_cur_object_name + str(object_postfixes[old_cur_object_name])
            for child in cur_object:
                if child.tag in ['joint', 'site']:
                    name, *postfix = child.attrib['name'].split(':')
                    child.attrib['name'] = ':'.join([new_cur_object_name, *postfix])
                if child.tag == 'geom':
                    child.attrib['rgba'] = object_color
                if child.tag == 'body':
                    for childs_child in child:
                        if childs_child.tag == 'geom':
                            childs_child.attrib['rgba'] = object_tint

            object_postfixes[old_cur_object_name] += 1
            cur_object.attrib['name'] = new_cur_object_name

            created_object_names.append(new_cur_object_name)
            worldbody.append(cur_object)
        
        # Fetch table size; needed for sampling of objects in scene
        table_pos, table_size = None, None
        for child in worldbody:
            if 'name' in child.attrib.keys() and child.attrib['name'] == 'table0':
                table_geom = child.find('geom')
                table_pos, table_size = child.attrib['pos'], table_geom.attrib['size']
                table_pos = np.array([float(coord) for coord in table_pos.split(' ')])
                table_size = np.array([float(half_length) for half_length in table_size.split(' ')])
                break

        # Change size of target cylinder accordingly to self.distance_threshold
        for child in worldbody:
            if 'name' in child.attrib.keys() and child.attrib['name'] == 'floor0':
                for childs_child in child:
                    if childs_child.attrib['name'] == 'target0':
                        radius, height = childs_child.attrib['size'].split(' ')
                        childs_child.attrib['size'] = str(self.distance_threshold) + ' ' + height

        env_xml_path = os.path.join(os.path.dirname(__file__), 'simulation_data', 'xmls', 'env.xml')
        if os.path.exists(env_xml_path):
            os.remove(env_xml_path)

        with open(env_xml_path, 'w') as f:
            xml_tree.write(f, encoding='unicode')
        return created_object_names, env_xml_path, table_pos, table_size
    
    def _prepare_object_positions(self, table_pos, table_size):
        # All center of masses must be at least on distance 0.07 from end of table
        # 0.07 - maximal distance from center of masses in created objects
        safe_margin = 0.08

        # All objects must be on distance of 0.2 from each other (in order to not overlap)
        safe_distance = 0.2

        # Ascension above table, in order for all objects to be above it
        safe_ascension = 0.03

        done = False
        while not done:
            done = True
            # Get uniform distribution in [-1., 1.]
            points = (self.np_random.uniform(size=(4, 2)) - 0.5) * 2
            # Convert uniform distribution into distribution with table size, inside safe zone
            points[:, 0] = points[:, 0] * (table_size[0] - safe_margin) + table_pos[0]
            points[:, 1] = points[:, 1] * (table_size[1] - safe_margin) + table_pos[1]
            
            # Check if all objects are on safe distance from eachother
            for i in range(points.shape[0]):
                for j in range(points.shape[0]):
                    if i != j:
                        if np.linalg.norm(points[i] - points[j]) < safe_distance:
                            done = False
        points = np.concatenate([points, np.ones([points.shape[0], 1]) * (table_pos[2] + table_size[2] + safe_ascension)],
                                axis = -1)
        
        # Sample goal position on necessary distance from each of objects
        done = False
        while not done:
            done = True
            goal = (self.np_random.uniform(size=(1, 2)) - 0.5) * 2
            goal[:, 0] = goal[:, 0] * (table_size[0] - safe_margin) + table_pos[0]
            goal[:, 1] = goal[:, 1] * (table_size[1] - safe_margin) + table_pos[1]

            for i in range(points.shape[0]):
                if np.linalg.norm(goal[0] - points[i][:2]) < self.distance_threshold + safe_margin:
                    done = False
            
        goal = np.concatenate([goal, np.zeros((1, 1))], axis = -1)
        return points, goal

    def _initialize_sim(self):
        created_object_names, xml_path, table_pos, table_size = self._create_multiobject_xml()
        model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(model, nsubsteps=self.n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array']
        }

        initial_qpos = copy.deepcopy(self.gripper_qpos)
        object_positions, goal = self._prepare_object_positions(table_pos=table_pos, table_size=table_size)
        for i, object_name in enumerate(created_object_names):
            initial_qpos['{}:joint'.format(object_name)] = [*object_positions[i], 1., 0., 0., 0.]

        self._env_setup(initial_qpos=initial_qpos)
        return created_object_names, goal
            
    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = gym_robotics_utils.robot_get_obs(self.sim)
        objects_description = {'object_pos': [], 'object_rel_pos': [], 'object_rot': [], 
                              'object_velp': [], 'object_velr': []}
        for name in self.created_object_names:
            objects_description['object_pos'].append(self.sim.data.get_site_xpos(name))
            # rotations
            objects_description['object_rot'].append(rotations.mat2euler(self.sim.data.get_site_xmat(name)))
            # velocities
            objects_description['object_velp'].append(self.sim.data.get_site_xvelp(name) * dt)
            objects_description['object_velr'].append(self.sim.data.get_site_xvelr(name) * dt)
            # gripper state
            objects_description['object_rel_pos'].append(objects_description['object_pos'][-1] - grip_pos)
            objects_description['object_velp'][-1] -= grip_velp
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        
        achieved_goal = np.stack(objects_description['object_pos'].copy(), axis = 0)
        assert len(achieved_goal.shape) == 2, 'Something wrong with achieved goal: expected 2 dimensions'

        for key in objects_description.keys():
            objects_description[key] = np.concatenate(objects_description[key], axis = 0)
        
        obs = np.concatenate([
            grip_pos, objects_description['object_pos'], objects_description['object_rel_pos'], gripper_state, 
            objects_description['object_rot'], objects_description['object_velp'], 
            objects_description['object_velr'], grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }
    
    def _is_success(self, achieved_goal, desired_goal):
        desired_goal = desired_goal.copy()[:, :2]
        for object_pos in achieved_goal[:, :2]:
            if np.linalg.norm(object_pos[0] - desired_goal[0], axis=-1) > self.distance_threshold:
                return False
        
        return True
    
    def compute_reward(self, achieved_goal, goal, info):
        total_objects_in_zone = 0
        goal = goal.copy()[0, :2]
        for object_pos in achieved_goal[:, :2]:
            total_objects_in_zone += (np.linalg.norm(object_pos - goal) < self.distance_threshold)
        reward = (total_objects_in_zone / len(achieved_goal))
        return reward - 1
    
    def reset(self):
        super(robot_env.RobotEnv, self).reset()
        self.created_object_names, self.goal = self._initialize_sim()
        obs = self._get_obs()
        return obs
    
    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        gym_robotics_utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

# ============= No change from fetch_env.PickAndPlaceEnv =============
    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        gym_robotics_utils.ctrl_set_action(self.sim, action)
        gym_robotics_utils.mocap_set_action(self.sim, action)

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        # TODO Why are they substract sites_offset[0] (only by x position)
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def render(self, mode='human', width=500, height=500):
        return super(MultipleFetchPickAndPlaceEnv, self).render(mode, width, height)

