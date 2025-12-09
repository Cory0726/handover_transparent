# Techman Robot Control (techmanpy)
`techmanpy` is an easy-to-use communication driver for **Techman robots** written in **Python**.
- [Reference](#reference)
- [System setup](#system-setup)
  - [Package installation](#package-installation)
- [Introduction](#introduction)
  - [Python async/await](#python-asyncawait)
- [Script](#script)
  - 
## Reference
- Github `techmanpy` : https://github.com/jvdtoorn/techmanpy

## System setup
### Package installation
```bash
pip install techmanpy
```

## Introduction
### [Python async/await](https://zhuanlan.zhihu.com/p/698683843)
- [Synchronous and Asynchronous](https://ithelp.ithome.com.tw/articles/10259764)
- **Coroutine** 就是在一個 **Thread** 裡, 通過 **Event Loop** 模擬出多個 **Thread** 並發的效果

## Script
### test_connection.py
To verify that your connection with the robot is all set-up.