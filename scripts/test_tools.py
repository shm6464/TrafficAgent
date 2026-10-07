import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.agent import _classify_file

files = [
    "01_行车组织管理办法_区间疏散与限速.md",
    "02_设施设备运行维护管理办法_检修周期.md",
    "03_CBTC信号系统原理.md",
    "04_车辆检修修程与周期.md",
    "05_运营突发事件应急预案.md",
    "06_牵引供电系统原理.md",
    "07_设备关键部位实时监控.md",
    "08_站台门屏蔽门系统.md",
    "09_CBTC与固定闭塞对比.md",
    "10_列车定位与测速技术.md",
    "11_供电系统运营技术规范.md",
    "12_列车再生制动与能量回馈.md",
    "13_信号系统维修分级与周期.md",
    "14_通信系统维修分级与周期.md",
    "15_火灾应急处置与消防安全.md",
    "16_应急预案体系与演练.md",
    "17_轨道设施养护维修.md",
    "18_自动售检票系统AFC.md",
    "19_系统总体概览.md",
]
for f in files:
    print(f"{_classify_file(f):<20} <- {f}")
