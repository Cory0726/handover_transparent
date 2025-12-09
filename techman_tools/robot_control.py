import asyncio
import techmanpy

async def query(robot_ip, data_list):
    async with techmanpy.connect_svr(robot_ip=robot_ip) as conn:
        params = await conn.get_values(data_list)
    return params

def query_tm_data(robot_ip):
    data_list = {'Coord_Robot_Flange', 'Joint_Angle', 'Project_Speed', 'End_DI0', 'End_DO0'}
    return asyncio.run(query(robot_ip, data_list))
