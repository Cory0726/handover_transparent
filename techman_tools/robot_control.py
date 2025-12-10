import asyncio
import techmanpy

async def query_params(robot_ip, data_list):
    async with techmanpy.connect_svr(robot_ip=robot_ip) as conn:
        params = await conn.get_values(data_list)
    return params

async def set_params(robot_ip, params):
    async with techmanpy.connect_svr(robot_ip=robot_ip) as conn:
        await conn.set_values(params)

async def motion_ptp(robot_ip, target_point, speed_perc, acc):
    async with techmanpy.connect_sct(robot_ip=robot_ip) as conn:
        await conn.move_to_point_ptp(target_point, speed_perc, acc)  # speed percentage(%), acceleration duration(ms)

async def motion_relative_ptp(robot_ip, target_point, speed_perc, acc):
    async with techmanpy.connect_sct(robot_ip=robot_ip) as conn:
        await conn.move_to_relative_point_ptp(target_point, speed_perc, acc, 0, True)  # speed percentage(%), acceleration duration(ms)

async def pick_and_place_async(robot_ip, pick_point, place_point, speed_perc, acc_dur, gripper_delay):
    # Move to pick point (Tool frame)
    async with techmanpy.connect_sct(robot_ip=robot_ip) as conn:
        trsct = conn.start_transaction()
        trsct.move_to_relative_point_ptp(pick_point, speed_perc, acc_dur, 0, True)
        trsct.set_queue_tag(1, wait_for_completion=True)
        await trsct.submit()
    # Close the gripper
    await set_params(robot_ip, {'End_DO0': True})
    await asyncio.sleep(gripper_delay)
    # Move to place point (Base frame)
    async with techmanpy.connect_sct(robot_ip=robot_ip) as conn:
        trsct = conn.start_transaction()
        trsct.move_to_point_ptp(place_point, speed_perc, acc_dur)
        trsct.set_queue_tag(2, wait_for_completion=True)
        await trsct.submit()
    # Open the gripper
    await set_params(robot_ip, {'End_DO0': False})
    await asyncio.sleep(gripper_delay)

class TMRobot:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.speed_perc = 0.60  # speed percentage(%)
        self.acc_dur = 200  # Acceleration duration(ms)
        self.gripper_delay = 2  # Unit : second

    def query_tm_data(self):
        data_list = {'Coord_Robot_Flange', 'End_DO0'}
        return asyncio.run(query_params(self.robot_ip, data_list))

    def gripper_close(self):
        params = {'End_DO0':True}
        asyncio.run(set_params(self.robot_ip,params))

    def gripper_open(self):
        params = {'End_DO0':False}
        asyncio.run(set_params(self.robot_ip,params))

    def move2target(self, target_point):
        asyncio.run(motion_ptp(self.robot_ip, target_point, self.speed_perc, self.acc_dur))

    def move2origin(self):
        # x, y, z, Rx, Ry, Rz (mm, deg)
        origin_point = [-400.1218, 12.36882, 636.417, -176.5101, 51.12951, 19.41987]
        # J1, J2, J3, J4, J5, J6 (deg)
        # origin_point = [196, -8, 68, -21, 89, 268]
        print(f'Origin Pose:\n\tTranslation: {origin_point[:3]} mm, Rotation: {origin_point[3:]} degrees')
        self.move2target(origin_point)

    def move2target_toolframe(self, target_point):
        asyncio.run(motion_relative_ptp(self.robot_ip, target_point, self.speed_perc, self.acc_dur))

    def pick_and_place(self, pick_point, place_point):
        asyncio.run(pick_and_place_async(self.robot_ip, pick_point, place_point, self.speed_perc, self.acc_dur, self.gripper_delay))