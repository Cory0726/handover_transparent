# Techman Robot Control (techmanpy)
`techmanpy` is an easy-to-use communication driver for **Techman robots** written in **Python**.
- [Reference](#reference)
- [Introduction](#introduction)
  - [Python async/await](#python-asyncawait)
- [System setup](#system-setup)
  - [Package installation](#package-installation)
- [Script](#script)


## Reference
- Github `techmanpy` : https://github.com/jvdtoorn/techmanpy

## Introduction
### [Python async/await](https://zhuanlan.zhihu.com/p/698683843)
- [Synchronous and Asynchronous](https://ithelp.ithome.com.tw/articles/10259764)
- **Coroutine** 就是在一個 **Thread** 裡, 通過 **Event Loop** 模擬出多個 **Thread** 並發的效果

## System setup
### Package installation
```bash
pip install techmanpy
```
### Robot preparation
1. TMFlow version >= 1.80
2. Setup Ethernet connection 
   - Go to the `System > Network setting` page, setup the **Static IP**.
3. Enable the TMFlow server
   1. Go to the `Settings > Connection` page and then to the **Ethernet Slave** tab. If it is currently enabled, disable it.
   2. Open the **Data Table Setting** overview
   3. On the top right, set **Communication Mode** to **STRING**
   4. Check all boxes in the **Predefined** list of items
   5. **Save** the configuration
   6. Now **enable** the ethernet slave
4. Put robot in **Listen mode**
   1. Create a new project
   2. Connect a **Listen** node to the **Start** node.
   3. Start the project

## Script
### test_connection.py
To verify that your connection with the robot is all set-up.
### test_recv_data.py
Create a simple socket client to listen to the TMFlow server broadcast. To verify that your robot setting is correctly.
### robot_control.py
- Ref **library** :
  - [TMFlow Server](https://github.com/jvdtoorn/techmanpy/wiki/TMFlow-Server)
  - [External Script](https://github.com/jvdtoorn/techmanpy/wiki/External-Script)