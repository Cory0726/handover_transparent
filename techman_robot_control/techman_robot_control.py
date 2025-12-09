import asyncio
import techmanpy

ROBOT_IP = "192.168.50.49"   # ← 改成你的 TM robot IP


async def read_robot_state():
    async with techmanpy.connect_svr(robot_ip=ROBOT_IP) as conn:

        # 在 callback 中解析 robot 狀態
        def handle_broadcast(data):
            """
            data 會是一個 dict，包含機器人的狀態資訊
            常見的 key：
                'JointAngle' → 六軸角度
                'TCP'        → TCP 位置 + 姿態
            """
            joint = data.get('JointAngle')
            tcp = data.get('TCP')

            if joint is not None:
                print(f"Joint Angles (deg): {joint}")

            if tcp is not None:
                # TCP format = [x, y, z, rx, ry, rz] (mm + degree)
                print(f"TCP Pose: {tcp}")

            print("-" * 50)

        # 把 callback 加入 SVR 監聽
        conn.add_broadcast_callback(handle_broadcast)

        # 保持連線（不然程式會結束）
        await conn.keep_alive()


if __name__ == "__main__":
    asyncio.run(read_robot_state())
