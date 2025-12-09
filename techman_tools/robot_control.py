import asyncio
import techmanpy

async def query_params(robot_ip, data_list):
    async with techmanpy.connect_svr(robot_ip=robot_ip) as conn:
        params = await conn.get_values(data_list)
    return params

async def set_params(robot_ip, params):
    async with techmanpy.connect_svr(robot_ip=robot_ip) as conn:
        await conn.set_values(params)

class TMRobot:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip

    def query_tm_data(self):
        data_list = {'Coord_Robot_Flange', 'Joint_Angle', 'Project_Speed', 'End_DI0', 'End_DO0'}
        return asyncio.run(query_params(self.robot_ip, data_list))

    def gripper_close(self):
        params = {'End_DO0':True}
        asyncio.run(set_params(self.robot_ip,params))

    def gripper_open(self):
        params = {'End_DO0':False}
        asyncio.run(set_params(self.robot_ip,params))
