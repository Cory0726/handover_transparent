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

class TMRobot:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip

    def query_tm_data(self):
        data_list = {'Coord_Robot_Tool', 'End_DO0'}
        return asyncio.run(query_params(self.robot_ip, data_list))

    def gripper_close(self):
        params = {'End_DO0':True}
        asyncio.run(set_params(self.robot_ip,params))

    def gripper_open(self):
        params = {'End_DO0':False}
        asyncio.run(set_params(self.robot_ip,params))


    # Wait for test
    def move2target(self, target_point):
        asyncio.run(motion_ptp(self.robot_ip, target_point, 30, 200))

    def move2target_toolframe(self, target_point):
        asyncio.run(motion_relative_ptp(self.robot_ip, target_point, 30, 200))

    def move2origin(self):
        # x, y, z, Rx, Ry, Rz (mm)
        origin_point = [-400.12, 12.37, 636.42, -176.51, 51.13, 19.42]
        # J1, J2, J3, J4, J5, J6 (deg)
        # origin_point = [196, -8, 68, -21, 89, 268]
        self.move2target(origin_point)