import asyncio
import techmanpy

async def test_connection(robot_ip):
  status = {'SCT': 'offline', 'SVR': 'offline', 'STA': 'offline'}

  # Check SVR connection (should be always active)
  try:
     async with techmanpy.connect_svr(robot_ip=robot_ip, conn_timeout=1) as conn:
        status['SVR'] = 'online'
        await conn.get_value('Robot_Model')
        status['SVR'] = 'connected'
  except techmanpy.TechmanException: pass

  # Check SCT connection (only active when inside listen node)
  try:
     async with techmanpy.connect_sct(robot_ip=robot_ip, conn_timeout=1) as conn:
        status['SCT'] = 'online'
        await conn.resume_project()
        status['SCT'] = 'connected'
  except techmanpy.TechmanException: pass

  # Check STA connection (only active when running project)
  try:
     async with techmanpy.connect_sta(robot_ip=robot_ip, conn_timeout=1) as conn:
        status['STA'] = 'online'
        await conn.is_listen_node_active()
        status['STA'] = 'connected'
  except techmanpy.TechmanException: pass

  # Print status
  def colored(status):
     if status == 'online': return f'\033[96m{status}\033[00m'
     if status == 'connected': return f'\033[92m{status}\033[00m'
     if status == 'offline': return f'\033[91m{status}\033[00m'

  print(f'SVR: {colored(status["SVR"])}, SCT: {colored(status["SCT"])}, STA: {colored(status["STA"])}')

if __name__ == '__main__':
    ROBOT_IP = '192.168.50.49'
    asyncio.run(test_connection(ROBOT_IP))
